#!/usr/bin/env python3

import threading

import rclpy
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import TwistStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

from robot_motion import ArmMover, wait_for
from common_interfaces_merlab.srv import SendJointTrajectory, SendJointTrajectoryPoint, SendPose, SendTwist

JOINT_NAMES = [f'joint_{i}' for i in range(1, 7)]


class ArmServices(Node):
    def __init__(self):
        super().__init__('arm_services')
        self.declare_parameter('ee_link', 'end_effector_link')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('speed', 0.1)
        self.declare_parameter('motion', 'cartesian')
        self.declare_parameter('cartesian_fallback', True)   # a service should not just refuse a reachable move
        self.declare_parameter('arm_controller', 'joint_trajectory_controller')
        self.declare_parameter('twist_controller', 'twist_controller')
        self.declare_parameter('twist_command_topic', '/twist_controller/commands')  # VERIFY on your robot
        self.declare_parameter('twist_frame_id', 'end_effector_link')                # VERIFY on your robot
        self.declare_parameter('twist_rate_hz', 30.0)

        clients_group = ReentrantCallbackGroup()
        self.mover = ArmMover(self, ee_link=self.get_parameter('ee_link').value,
                              base_frame=self.get_parameter('base_frame').value,
                              speed=self.get_parameter('speed').value,
                              motion=self.get_parameter('motion').value,
                              cartesian_fallback=self.get_parameter('cartesian_fallback').value,
                              callback_group=clients_group)
        self.traj_client = ActionClient(self, FollowJointTrajectory,
                                        '/joint_trajectory_controller/follow_joint_trajectory',
                                        callback_group=clients_group)
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller',
                                                callback_group=clients_group)
        self.twist_pub = self.create_publisher(TwistStamped, self.get_parameter('twist_command_topic').value, 10)
        self._twist_timer = None
        self._twist_msg = None

        self.create_service(SendPose, 'send_pose', self.on_send_pose)
        self.create_service(SendJointTrajectoryPoint, 'send_joint_trajectory_point', self.on_send_point)
        self.create_service(SendJointTrajectory, 'send_joint_trajectory', self.on_send_trajectory)
        self.create_service(SendTwist, 'send_twist', self.on_send_twist)
        self.get_logger().info('Ready: /send_pose, /send_joint_trajectory_point, /send_joint_trajectory, '
                               '/send_twist (send_twist is best-effort; verify before relying on it)')

    def on_send_pose(self, request, response):
        p = request.pose
        pose = [p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        response.success = self.mover.move_to(pose)
        return response

    def on_send_point(self, request, response):
        response.success = self._run_trajectory([request.goal_point])
        return response

    def on_send_trajectory(self, request, response):
        traj = request.goal_points
        names = list(traj.joint_names) if traj.joint_names else JOINT_NAMES
        response.success = self._run_trajectory(traj.points, names)
        return response

    def _run_trajectory(self, points, joint_names=None):
        if not self.traj_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('joint_trajectory_controller action server not available')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names or JOINT_NAMES
        goal.trajectory.points = list(points)
        handle = wait_for(self.traj_client.send_goal_async(goal), timeout=10.0)
        if handle is None or not handle.accepted:
            self.get_logger().error('Trajectory goal rejected')
            return False
        result = wait_for(handle.get_result_async(), timeout=120.0)
        ok = result is not None and result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if not ok:
            self.get_logger().error(f'Trajectory execution failed: {None if result is None else result.result}')
        return ok

    # best-effort/unverified, see the top of this file
    def on_send_twist(self, request, response):
        t = request.twist
        zero = all(abs(v) < 1e-6 for v in (t.linear.x, t.linear.y, t.linear.z,
                                           t.angular.x, t.angular.y, t.angular.z))
        if zero:
            self._stop_twist()
            response.success, response.message = True, 'stopped, switched back to arm_controller'
            return response
        if not self._switch(activate=self.get_parameter('twist_controller').value,
                            deactivate=self.get_parameter('arm_controller').value):
            response.success = False
            response.message = ('could not activate twist_controller via /controller_manager/'
                                'switch_controller -- see the node log')
            return response
        self._start_twist(t)
        response.success = True
        response.message = ('streaming twist on ' + self.get_parameter('twist_command_topic').value +
                            ' -- topic/message type UNVERIFIED, confirm before trusting this')
        return response

    def _switch(self, activate, deactivate):
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/controller_manager/switch_controller not available')
            return False
        req = SwitchController.Request()
        req.activate_controllers = [activate]
        req.deactivate_controllers = [deactivate]
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        result = wait_for(self.switch_client.call_async(req), timeout=10.0)
        return result is not None and result.ok

    def _start_twist(self, twist):
        self._twist_msg = TwistStamped()
        self._twist_msg.header.frame_id = self.get_parameter('twist_frame_id').value
        self._twist_msg.twist = twist
        if self._twist_timer is None:
            period = 1.0 / max(1.0, self.get_parameter('twist_rate_hz').value)
            self._twist_timer = self.create_timer(period, self._publish_twist)

    def _publish_twist(self):
        if self._twist_msg is not None:
            self._twist_msg.header.stamp = self.get_clock().now().to_msg()
            self.twist_pub.publish(self._twist_msg)

    def _stop_twist(self):
        if self._twist_timer is not None:
            self._twist_timer.cancel()
            self._twist_timer = None
        zero = TwistStamped()
        zero.header.frame_id = self.get_parameter('twist_frame_id').value
        for _ in range(3):        # a few zero commands so the controller actually sees a stop
            zero.header.stamp = self.get_clock().now().to_msg()
            self.twist_pub.publish(zero)
        self._switch(activate=self.get_parameter('arm_controller').value,
                    deactivate=self.get_parameter('twist_controller').value)


def main():
    rclpy.init()
    node = ArmServices()
    executor = MultiThreadedExecutor()   
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()