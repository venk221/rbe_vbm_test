#!/usr/bin/env python3
# Move the arm to each viewpoint in views.yaml and save a cloud there, in base_link.
# Usage:
#   python3 capture_views.py                                   # views from views.yaml
#   python3 capture_views.py --pose x y z qx qy qz qw --pose x y z qx qy qz qw   

import argparse
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from helpers.geometry import apply_transform, crop_box, save_cloud, transform_to_matrix, voxel_downsample
from helpers.robot_motion import ArmMover, parse_pose
from helpers.ros_clouds import msg_to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
RELIABLE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)


class ViewCapturer(Node):
    def __init__(self, config):
        super().__init__('multiview_capture')
        self.cfg = config
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.collecting = False            
        self.frames = []
        self.received = 0                 
        self.create_subscription(PointCloud2, config['cloud_topic'], self.on_cloud, RELIABLE_QOS)

    def on_cloud(self, msg):
        
        self.received += 1
        if self.collecting and len(self.frames) < self.cfg['frames_per_view']:
            self.frames.append(msg)

    # fails early, before moving the robot, if base_link -> camera isn't connected
    def wait_for_tf(self, timeout=15.0):
        child = self.cfg['hand_eye']['child']
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.tf_buffer.can_transform(self.cfg['base_frame'], child, Time()):
                return True
            time.sleep(0.2)
        self.get_logger().error(f"No TF {self.cfg['base_frame']} -> {child}: "
                                f'is bringup_robot.launch.py (and bringup_camera.launch.py) running?')
        return False

    def capture(self):
        self.get_logger().info(f"Waiting {self.cfg['settle_time']:.1f} s for the arm to settle ...")
        time.sleep(self.cfg['settle_time'])
        self.frames = []
        received_before = self.received
        self.collecting = True
        deadline = time.time() + 10.0
        while len(self.frames) < self.cfg['frames_per_view'] and time.time() < deadline:
            time.sleep(0.02)
        self.collecting = False
        if not self.frames:
            if self.received == 0:
                why = 'no cloud has arrived since this script started: is the camera publishing? ' \
                      f"(check: ros2 topic hz {self.cfg['cloud_topic']})"
            elif self.received == received_before:
                why = 'clouds arrived earlier but stopped: the camera probably dropped off USB during the move ' \
                      '(look for "No such device" in the launch terminal)'
            else:
                why = 'clouds arrived but were not stored (unexpected; please report this)'
            self.get_logger().error(f"No clouds captured on {self.cfg['cloud_topic']}: {why}")
            return None
        self.get_logger().info(f'Got {len(self.frames)} clouds')

        all_points, all_colors, T_first = [], [], None
        for msg in self.frames:
            try:   
                tf = self.tf_buffer.lookup_transform(self.cfg['base_frame'], msg.header.frame_id,
                                                     Time.from_msg(msg.header.stamp),
                                                     timeout=Duration(seconds=0.5))
            except TransformException:
                try:   # or the latest one
                    tf = self.tf_buffer.lookup_transform(self.cfg['base_frame'], msg.header.frame_id, Time(),
                                                         timeout=Duration(seconds=1.0))
                except TransformException as e:
                    self.get_logger().error(f"TF {msg.header.frame_id} -> {self.cfg['base_frame']} failed: {e}")
                    return None
            T = transform_to_matrix(tf.transform)
            T_first = T if T_first is None else T_first
            points, colors = msg_to_numpy(msg)
            max_depth = self.cfg.get('max_depth')
            if max_depth:        
                near = points[:, 2] <= max_depth
                points, colors = points[near], (colors[near] if colors is not None else None)
            points, colors = crop_box(apply_transform(points, T), colors, self.cfg['workspace'])
            all_points.append(points)
            if colors is not None:
                all_colors.append(colors)

        points = np.concatenate(all_points)
        colors = np.concatenate(all_colors) if len(all_colors) == len(all_points) else None
        # points, colors = voxel_downsample(points, self.cfg['view_voxel'], colors)  
        return points, colors, T_first


def main():
    parser = argparse.ArgumentParser(description='Multi-view point cloud capture')
    parser.add_argument('--config', default=os.path.join(HERE, 'config', 'views.yaml'), help='per-robot YAML')
    parser.add_argument('--pose', nargs=7, type=float, action='append', metavar=('X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW'),
                        help='end-effector pose in base_link; repeat once per view (overrides views in the YAML)')
    parser.add_argument('--settle', type=float, help='seconds to wait at each view (overrides settle_time)')
    args, _ = parser.parse_known_args()     

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.settle is not None:
        cfg['settle_time'] = args.settle
    raw_views = args.pose if args.pose else (cfg.get('views') or [])
    try:
        views = [parse_pose(v, f'view {k}') for k, v in enumerate(raw_views)]
        home = parse_pose(cfg['home'], 'home') if cfg.get('home') is not None else None
    except ValueError as e:
        print(e)
        return 1
    if len(views) < 2:
        print(f'Give at least two views: --pose ... --pose ..., or a views list in {args.config}.')
        return 1

    rclpy.init(args=None)
    node = ViewCapturer(cfg)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    log = node.get_logger()
    mover = ArmMover(node, ee_link=cfg['ee_link'], base_frame=cfg['base_frame'], speed=cfg['speed'],
                     motion=cfg.get('motion', 'cartesian'), cartesian_fallback=cfg.get('cartesian_fallback', False))

    session = os.path.join(HERE, 'captures', datetime.now().strftime('%Y%m%d_%H%M%S'))
    ok = node.wait_for_tf()
    if ok:
        os.makedirs(session)
    try:
        for k, target in enumerate(views if ok else []):
            log.info(f'View {k}: moving to pose {np.round(target, 3).tolist()} ...')
            moved = mover.move_to(target)
            if not moved:
                ok = False
                break
            result = node.capture()
            if result is None:
                ok = False
                break
            points, colors, T = result
            save_cloud(os.path.join(session, f'view_{k}.npz'), points, colors,
                       T_base_camera=T, target=np.array(target))
            log.info(f'View {k}: saved {len(points)} points (camera at '
                     f"x={T[0, 3]:.3f} y={T[1, 3]:.3f} z={T[2, 3]:.3f} in {cfg['base_frame']})")
        if home is not None:
            log.info('Returning home ...')
            mover.move_to(home)
    finally:
        executor.shutdown()      
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()

    if ok:
        print(f'\nCapture finished: {session}\nNext: python3 {os.path.join(HERE, "merge_views.py")}')
    else:
        print('\nCapture FAILED (see errors above).')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())