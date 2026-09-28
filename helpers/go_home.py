#!/usr/bin/env python3
#python3 go_home.py [path/to/views.yaml]
import os
import sys
import threading
import time

import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from robot_motion import ArmMover, parse_pose

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'config', 'views.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get('home') is None:
        print(f'No home pose set in {config_path}; not moving.')
        return 0
    try:
        home = parse_pose(cfg['home'], 'home')
    except ValueError as e:
        print(e)
        return 1

    rclpy.init()
    node = Node('go_home')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    mover = ArmMover(node, ee_link=cfg['ee_link'], base_frame=cfg['base_frame'], speed=cfg['speed'],
                     motion=cfg.get('motion', 'cartesian'), cartesian_fallback=cfg.get('cartesian_fallback', False))
    ok = False
    try:
        if mover.wait_until_ready(timeout=120.0):
            for attempt in range(1, 4):        
                node.get_logger().info(f'Moving home (attempt {attempt}) ...')
                if mover.move_to(home):
                    ok = True
                    break
                time.sleep(3.0)
    finally:
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    print('Home reached.' if ok else 'Could not reach home (see errors above).')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())