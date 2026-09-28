# A pose is [x, y, z, qx, qy, qz, qw]: position (m) + orientatin in the base frame.
# motion = 'cartesian'  
# motion = 'rrtconnect' 
import time

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive

ERROR_HINTS = {-31: ' (no IK solution: pose unreachable)', -10: ' (start state invalid, e.g. joint_3 limit)',
               -1: ' (planning failed)', -4: ' (execution failed)'}


def parse_pose(entry, what='pose'):
    if isinstance(entry, dict) and 'pose' in entry:
        entry = entry['pose']
    if isinstance(entry, (list, tuple)) and len(entry) == 7:
        return [float(v) for v in entry]
    raise ValueError(f'{what}: expected a pose [x, y, z, qx, qy, qz, qw] (7 numbers), got {entry!r}')


def list_to_pose(p):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p[:3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in p[3:])
    return pose


def max_joint_jump(trajectory):
    points = trajectory.joint_trajectory.points
    jump = 0.0
    for a, b in zip(points, points[1:]):
        jump = max(jump, max(abs(q1 - q0) for q0, q1 in zip(a.positions, b.positions)))
    return jump


def wait_for(future, timeout):
    start = time.time()
    while not future.done():
        if time.time() - start > timeout:
            return None
        time.sleep(0.01)
    return future.result()


class ArmMover:
    def __init__(self, node, group='arm', ee_link='end_effector_link', base_frame='base_link', speed=0.1,
                 motion='cartesian', cartesian_fallback=False, max_step=0.005, max_jump=0.5, callback_group=None):
        self.node, self.group, self.ee_link, self.base_frame, self.speed = node, group, ee_link, base_frame, speed
        self.motion, self.cartesian_fallback = motion, cartesian_fallback
        self.max_step = max_step          # m between interpolated points on a straight-line path
        self.max_jump = max_jump       
        self.move_client = ActionClient(node, MoveGroup, 'move_action', callback_group=callback_group)
        self.exec_client = ActionClient(node, ExecuteTrajectory, 'execute_trajectory', callback_group=callback_group)
        self.cart_client = node.create_client(GetCartesianPath, 'compute_cartesian_path',
                                              callback_group=callback_group)
        self.log = node.get_logger()

    def wait_until_ready(self, timeout=120.0):
        deadline = time.time() + timeout
        ok = (self.move_client.wait_for_server(timeout_sec=timeout)
              and self.exec_client.wait_for_server(timeout_sec=max(1.0, deadline - time.time()))
              and self.cart_client.wait_for_service(timeout_sec=max(1.0, deadline - time.time())))
        if not ok:
            self.log.error('MoveIt services not available: is the MoveIt launch running?')
        return ok

    def move_to(self, pose):
        if self.motion == 'cartesian':
            if self.move_cartesian(pose):
                return True
            if not self.cartesian_fallback:
                return False
            self.log.warn('Straight line not possible; falling back to RRTConnect')
        return self.move_rrtconnect(pose)

    def move_cartesian(self, pose):
        if not self.wait_until_ready(timeout=10.0):
            return False
        req = GetCartesianPath.Request()
        req.header.frame_id, req.group_name, req.link_name = self.base_frame, self.group, self.ee_link
        req.start_state.is_diff = True
        req.waypoints = [list_to_pose(pose)]
        req.max_step = self.max_step
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = req.max_acceleration_scaling_factor = self.speed
        res = wait_for(self.cart_client.call_async(req), timeout=15.0)
        if res is None:
            self.log.error('compute_cartesian_path did not answer')
            return False
        if res.fraction < 0.999:
            self.log.error(f'Straight line only {res.fraction:.0%} feasible (joint limit, singularity or '
                           'collision on the way); not moving')
            return False
        jump = max_joint_jump(res.solution)
        if jump > self.max_jump:
            self.log.error(f'Straight line needs a sudden joint jump of {jump:.2f} rad (wrist/elbow flip); '
                           'not moving')
            return False
        return self._check(self._send(self.exec_client, ExecuteTrajectory.Goal(trajectory=res.solution)))

    def move_rrtconnect(self, pose):
        target = list_to_pose(pose)
        position = PositionConstraint()
        position.header.frame_id, position.link_name, position.weight = self.base_frame, self.ee_link, 1.0
        position.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005]))
        position.constraint_region.primitive_poses.append(target)
        orientation = OrientationConstraint()
        orientation.header.frame_id, orientation.link_name, orientation.weight = self.base_frame, self.ee_link, 1.0
        orientation.orientation = target.orientation
        orientation.absolute_x_axis_tolerance = 0.02
        orientation.absolute_y_axis_tolerance = 0.02
        orientation.absolute_z_axis_tolerance = 0.02

        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.log.error('move_action not available: is the MoveIt launch running?')
            return False
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name, r.pipeline_id, r.planner_id = self.group, 'ompl', 'RRTConnectkConfigDefault'
        r.allowed_planning_time, r.num_planning_attempts = 5.0, 5
        r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = self.speed
        r.start_state.is_diff = True
        r.goal_constraints.append(Constraints(position_constraints=[position],
                                              orientation_constraints=[orientation]))
        goal.planning_options.plan_only = False
        return self._check(self._send(self.move_client, goal))

    def _send(self, client, goal):
        handle = wait_for(client.send_goal_async(goal), timeout=10.0)
        if handle is None or not handle.accepted:
            self.log.error('MoveIt rejected the goal')
            return None
        response = wait_for(handle.get_result_async(), timeout=120.0)
        return None if response is None else response.result

    def _check(self, result):
        code = None if result is None else result.error_code.val
        if code != 1:
            self.log.error(f'Motion failed, MoveIt error code {code}{ERROR_HINTS.get(code, "")}')
        return code == 1
