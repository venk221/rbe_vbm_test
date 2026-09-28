#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from shape_msgs.msg import SolidPrimitive

# Target end-effector pose in base_link
# ros2 run tf2_ros tf2_echo base_link end_effector_link
X, Y, Z = 0.238, -0.085, 0.460          
QX, QY, QZ, QW = -0.014, 0.999, -0.020, -0.021  # Rotation (quaternion)

FRAME = 'base_link'
EE_LINK = 'end_effector_link'


def main():
    rclpy.init()
    node = rclpy.create_node('go_to_pose')

    target = Pose()
    target.position.x, target.position.y, target.position.z = X, Y, Z
    target.orientation.x, target.orientation.y = QX, QY
    target.orientation.z, target.orientation.w = QZ, QW

    # Position goal
    position = PositionConstraint()
    position.header.frame_id = FRAME
    position.link_name = EE_LINK
    position.constraint_region.primitives.append(
        SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005]))
    position.constraint_region.primitive_poses.append(target)
    position.weight = 1.0

    # Orientation goal
    orientation = OrientationConstraint()
    orientation.header.frame_id = FRAME
    orientation.link_name = EE_LINK
    orientation.orientation = target.orientation
    orientation.absolute_x_axis_tolerance = 0.01
    orientation.absolute_y_axis_tolerance = 0.01
    orientation.absolute_z_axis_tolerance = 0.01
    orientation.weight = 1.0

    goal = MoveGroup.Goal()
    goal.request.group_name = 'arm'
    goal.request.pipeline_id = 'ompl'
    goal.request.planner_id = 'RRTConnectkConfigDefault'
    goal.request.allowed_planning_time = 5.0
    goal.request.num_planning_attempts = 5
    goal.request.max_velocity_scaling_factor = 0.1       
    goal.request.max_acceleration_scaling_factor = 0.1
    goal.request.start_state.is_diff = True              
    goal.request.goal_constraints.append(
        Constraints(position_constraints=[position], orientation_constraints=[orientation]))
    goal.planning_options.plan_only = False              

    client = ActionClient(node, MoveGroup, 'move_action')
    client.wait_for_server()
    goal_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, goal_future)
    result_future = goal_future.result().get_result_async()
    rclpy.spin_until_future_complete(node, result_future)

    code = result_future.result().result.error_code.val
    node.get_logger().info('Reached pose' if code == 1 else f'Failed, MoveIt error code {code}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()