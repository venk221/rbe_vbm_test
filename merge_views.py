#!/usr/bin/env python3
# python3 merge_views.py path to views

import argparse
import glob
import os
import sys

import numpy as np
from pcl_python_merlab import PointCloud

from helpers.geometry import load_cloud, rotation_angle_deg, save_cloud
from helpers.icp import alignment_error, icp_point_to_plane

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CORRECTION_M = 0.02     
MAX_CORRECTION_DEG = 5.0     
MIN_FITNESS = 0.3           
OUTLIER_NEIGHBORS = 20       
OUTLIER_STDDEV = 2.0        


def remove_outliers(points):
    if len(points) <= OUTLIER_NEIGHBORS:
        return points
    return PointCloud(points).statistical_outlier_removal(
        mean_k=OUTLIER_NEIGHBORS, stddev_multiplier=OUTLIER_STDDEV).xyz


# ICP
def refine_with_icp(source, target, voxel, max_distance):
    src_ds = PointCloud(source).voxel_grid(leaf_size=voxel).xyz
    tgt_ds = PointCloud(target).voxel_grid(leaf_size=voxel).xyz
    fitness0, rmse0 = alignment_error(src_ds, tgt_ds, max_distance)
    T = np.eye(4)
    for radius in (4 * max_distance, 2 * max_distance, max_distance):
        moved = PointCloud(src_ds).transform(T).xyz
        T_step, fitness, rmse = icp_point_to_plane(moved, tgt_ds, max_distance=radius)
        T = T_step @ T
    moved = PointCloud(src_ds).transform(T).xyz
    fitness, rmse = alignment_error(moved, tgt_ds, max_distance)
    shift_mm, angle = np.linalg.norm(T[:3, 3]) * 1000, rotation_angle_deg(T[:3, :3])
    report = (f'overlap {fitness0:.0%} -> {fitness:.0%}, RMS {rmse0 * 1000:.1f} -> {rmse * 1000:.1f} mm, '
              f'correction {shift_mm:.1f} mm / {angle:.2f} deg')
    if shift_mm > MAX_CORRECTION_M * 1000 or angle > MAX_CORRECTION_DEG or fitness < MIN_FITNESS:
        return np.eye(4), report + '  -> REJECTED, keeping TF-only alignment'
    return T, report


# concatenate + voxel-downsample
def merge(views, voxel):
    cloud = PointCloud(views[0])
    for points in views[1:]:
        cloud = cloud.concatenate(PointCloud(points))
    return cloud.voxel_grid(leaf_size=voxel).xyz


def publish_forever(clouds, frame_id):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from helpers.ros_clouds import numpy_to_msg

    rclpy.init()
    node = Node('multiview_merge')
    publishers = {name: node.create_publisher(PointCloud2, f'/multiview/{name}', 1) for name in clouds}

    def publish():
        stamp = node.get_clock().now().to_msg()
        for name, points in clouds.items():
            publishers[name].publish(numpy_to_msg(points, None, frame_id, stamp))

    node.create_timer(1.0, publish)
    print('Publishing ' + ', '.join(f'/multiview/{n}' for n in clouds) + f' in {frame_id}. Ctrl-C to stop.')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Merge saved views (already in base_link) into one cloud.')
    parser.add_argument('session', nargs='?', help='capture folder (default: newest in captures/)')
    parser.add_argument('--voxel', type=float, default=0.004, help='output voxel size in m (default 0.004)')
    parser.add_argument('--icp-voxel', type=float, default=0.004, help='voxel size used inside ICP (m)')
    parser.add_argument('--icp-distance', type=float, default=0.005,
                        help='final ICP correspondence distance in m; coarse stages use 4x and 2x (default 0.005)')
    parser.add_argument('--no-icp', action='store_true', help='TF-only merge')
    parser.add_argument('--keep-outliers', action='store_true', help='skip statistical outlier removal')
    parser.add_argument('--no-publish', action='store_true', help='save files and exit')
    parser.add_argument('--frame', default='base_link')
    args = parser.parse_args()

    session = args.session or max(glob.glob(os.path.join(HERE, 'captures', '*')), default=None)
    files = sorted(glob.glob(os.path.join(session or '', 'view_*.npz')),
                   key=lambda f: int(os.path.basename(f)[5:-4]))
    if len(files) < 2:
        print(f'Need at least two view_*.npz files in {session}. Run capture_views.py first.')
        return 1
    views = [load_cloud(f)[0] for f in files]
    print(f'Loaded {len(views)} views from {session}: ' + ', '.join(str(len(p)) for p in views) + ' points')
    if not args.keep_outliers:
        views = [remove_outliers(p) for p in views]
        print('After outlier removal: ' + ', '.join(str(len(p)) for p in views) + ' points')

    outputs = {f'view_{k}': view for k, view in enumerate(views)}

    if not args.no_icp:
        target_points = views[0]
        aligned = [views[0]]
        for k, points in enumerate(views[1:], start=1):
            T, report = refine_with_icp(points, target_points, args.icp_voxel, args.icp_distance)
            print(f'ICP view {k} -> view 0: {report}')
            aligned.append(PointCloud(points).transform(T).xyz)
        final = merge(aligned, args.voxel)
    else:
        final = merge(views, args.voxel)

    outputs['merged'] = final
    save_cloud(os.path.join(session, 'merged.npz'), final)
    print(f'Saved merged cloud ({len(final)} points) to {session}/merged.npz')

    if not args.no_publish:
        publish_forever(outputs, args.frame)
    return 0


if __name__ == '__main__':
    sys.exit(main())