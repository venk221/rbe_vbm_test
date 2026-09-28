#!/usr/bin/env python3
#python3 joint_to_poses.py [path_to/views.yaml]
import os
import sys
import threading

import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))   
from helpers.robot_motion import ArmMover, parse_target


def fmt(pose):
    return '[' + ', '.join(f'{v:.5f}' for v in pose[:3]) + ', ' + ', '.join(f'{v:.6f}' for v in pose[3:]) + ']'


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'config', 'views.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rclpy.init()
    node = Node('joints_to_poses')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    mover = ArmMover(node, ee_link=cfg['ee_link'], base_frame=cfg['base_frame'])

    def to_pose(entry):
        kind, target = parse_target(entry)
        return target if kind == 'pose' else mover.forward_kinematics(target)

    try:
        views = [to_pose(v) for v in (cfg.get('views') or [])]
        home = to_pose(cfg['home']) if cfg.get('home') is not None else None
    finally:
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()

    if any(p is None for p in views) or (cfg.get('home') is not None and home is None):
        print('Conversion failed (see errors above).')
        return 1
    print(f'\n# Paste into {config_path}, replacing the views and home entries:\n')
    print('views:')
    for k, pose in enumerate(views):
        print(f'  - pose: {fmt(pose)}     # view {k}')
    if home is not None:
        print(f'\nhome:\n  pose: {fmt(home)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())