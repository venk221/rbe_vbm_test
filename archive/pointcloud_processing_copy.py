#!/usr/bin/env python3
# Takes the merged cloud, finds whatever's sitting on the table, and works out a grasp for it.
# Usage:
#   python3 pointcloud_processing.py
#   python3 pointcloud_processing.py --ros-args -p max_width:=0.085

import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
from pcl_python_merlab import KdTree, PointCloud
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from grasp_synthesis import approach_direction, grasp_rotation, matrix_to_quaternion
from helpers.ros_clouds import numpy_to_msg

HERE = os.path.dirname(os.path.abspath(__file__))
RELIABLE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
COLORS = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                   [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128]], np.uint8)


def workspace_from_views_yaml():
    try:
        with open(os.path.join(HERE, 'config', 'views.yaml')) as f:
            ws = yaml.safe_load(f)['workspace']
        return [float(v) for v in ws['x'] + ws['y'] + ws['z']]
    except (OSError, KeyError, TypeError):
        return [0.20, 0.70, -0.30, 0.30, -0.03, 0.30]


def cloud_from_msg(msg):
    xyz = point_cloud2.read_points_numpy(
        msg, field_names=('x', 'y', 'z'), skip_nans=False,
    ).reshape(-1, 3)
    return PointCloud(xyz[np.isfinite(xyz).all(axis=1)].astype(np.float32))


def transform_cloud(cloud, transform):
    q = transform.transform.rotation
    t = transform.transform.translation
    quaternion = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError('TF rotation must be a finite, nonzero quaternion')
    x, y, z, w = quaternion / norm
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (t.x, t.y, t.z)
    return cloud.transform(matrix)


# workspace crop box (just three PassThrough filters chained together)
def crop_workspace(cloud, workspace):
    xmin, xmax, ymin, ymax, zmin, zmax = workspace
    cloud = cloud.pass_through(axis='x', minimum=xmin, maximum=xmax)
    cloud = cloud.pass_through(axis='y', minimum=ymin, maximum=ymax)
    cloud = cloud.pass_through(axis='z', minimum=zmin, maximum=zmax)
    return cloud


# find the table plane
def segment_table(cloud, distance_threshold, max_iterations, max_tilt_deg, attempts, log):
    for _ in range(attempts):
        if len(cloud) < 3:
            return None, None
        inliers, coefficients = cloud.segment_plane(
            distance_threshold=distance_threshold, max_iterations=max_iterations)
        if not inliers:
            return None, None
        plane = np.asarray(coefficients, dtype=np.float64)
        plane /= np.linalg.norm(plane[:3])          
        if plane[2] < 0:
            plane = -plane                         
        tilt_deg = np.degrees(np.arccos(np.clip(plane[2], -1.0, 1.0)))
        if tilt_deg <= max_tilt_deg:
            return plane, inliers
        log(f'Skipping a plane tilted {tilt_deg:.0f} deg ({len(inliers)} pts): not the table')
        cloud = cloud.extract(inliers, negative=True)
    return None, None


# segment objects on the table
def segment_objects(cloud, plane, table_indices, min_height, z_max):
    objects = cloud.extract(table_indices, negative=True)
    if len(objects) == 0:
        return objects
    table_z = -plane[3] / plane[2]                  
    return objects.pass_through(axis='z', minimum=table_z + min_height, maximum=z_max)


# split the leftover points into clusters per object
def cluster_objects(objects, tolerance, min_size, max_size):
    if len(objects) == 0:
        return []
    groups = objects.euclidean_clusters(tolerance=tolerance, min_size=min_size, max_size=max_size)
    clusters = [objects.extract(indices) for indices in groups]
    clusters.sort(key=len, reverse=True)
    return clusters


# centroid + surface normals for one object
def centroid_and_normals(cluster, normal_radius):
    centroid = cluster.xyz.astype(np.float64).mean(axis=0)
    normals = cluster.estimate_normals(radius=normal_radius, viewpoint=centroid.tolist())
    valid = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 1e-8)
    cluster = cluster.extract(np.flatnonzero(valid).tolist())
    normals = normals[valid]
    return centroid, cluster, normals


# find a pair of points the gripper could actually pinch. Among feasible pairs, prefers the
# shorter-than-average ones -- pinching an object's short edge is more stable than spanning its
# long edge, and eats up less of the gripper's travel -- then picks the highest one of those.
def find_grasp_pair(cluster, normals, mu, min_width, max_width):
    points = cluster.xyz.astype(np.float64)
    cos_alpha = np.cos(np.arctan(mu))
    tree = KdTree(cluster)
    candidates = []           # every feasible pair: (i, j, width, mean_z)
    feasible = 0

    for i in range(len(points)):
        neighbours, _ = tree.radius_search(points[i], max_width)
        j_candidates = np.asarray(neighbours, dtype=np.int64)
        j_candidates = j_candidates[j_candidates > i]
        if len(j_candidates) == 0:
            continue
        v = points[j_candidates] - points[i]
        width = np.linalg.norm(v, axis=1)
        d = v / np.maximum(width, 1e-12)[:, None]
        ok = ((width >= min_width)
              & (d @ normals[i] >= cos_alpha)
              & (np.einsum('ij,ij->i', normals[j_candidates], -d) >= cos_alpha))
        if not ok.any():
            continue
        feasible += int(ok.sum())
        mean_z = (points[i, 2] + points[j_candidates[ok], 2]) / 2.0
        for j, w, z in zip(j_candidates[ok], width[ok], mean_z):
            candidates.append((i, int(j), float(w), float(z)))

    if not candidates:
        return None, feasible
    avg_width = sum(c[2] for c in candidates) / len(candidates)
    shorter = [c for c in candidates if c[2] <= avg_width]      # grasp the short edge, not the long one
    pool = shorter or candidates
    i, j, _, _ = max(pool, key=lambda c: c[3])                   # of those, still prefer the highest one
    p1, p2, n1, n2 = points[i], points[j], normals[i], normals[j]
    width = float(np.linalg.norm(p2 - p1))
    close_dir = (p2 - p1) / width
    angles = (float(np.degrees(np.arccos(np.clip(n1 @ close_dir, -1.0, 1.0)))),
              float(np.degrees(np.arccos(np.clip(n2 @ -close_dir, -1.0, 1.0)))))
    return dict(p1=p1, p2=p2, n1=n1, n2=n2, close_dir=close_dir, width=width, angles=angles), feasible


def _arrowhead(p, n, length):
    tip = p + length * n
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp = np.cross(n, up)
    perp /= np.linalg.norm(perp)
    back = tip - 0.3 * length * n
    return tip, back + 0.15 * length * perp, back - 0.15 * length * perp


# RViz: draw each object's centroid + normals
def normals_marker_array(clusters_data, frame_id, stamp, length, stride):
    markers = [Marker(action=Marker.DELETEALL)]
    for k, data in enumerate(clusters_data):
        centroid, cluster, normals = data['centroid'], data['cluster'], data['normals']
        color = COLORS[k % len(COLORS)].astype(np.float64) / 255.0
        rgba = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
        header = Header(frame_id=frame_id, stamp=stamp)
        base = 2 * k

        lines = Marker(header=header, ns='normals', id=base, type=Marker.LINE_LIST,
                       action=Marker.ADD, scale=Vector3(x=0.0015), color=rgba)
        for p, n in zip(cluster.xyz[::stride], normals[::stride]):
            tip, barb1, barb2 = _arrowhead(p, n, length)
            for a, b in ((p, tip), (tip, barb1), (tip, barb2)):
                lines.points.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
                lines.points.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
        markers.append(lines)

        markers.append(Marker(
            header=header, ns='centroids', id=base + 1, type=Marker.SPHERE, action=Marker.ADD,
            pose=Pose(position=Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]))),
            scale=Vector3(x=0.012, y=0.012, z=0.012), color=rgba,
        ))
    return MarkerArray(markers=markers)


# RViz: draw just the ONE grasp we actually picked
def grasp_marker_array(pair, position, approach, frame_id, stamp):
    header = Header(frame_id=frame_id, stamp=stamp)
    yellow = ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0)
    green = ColorRGBA(r=0.2, g=1.0, b=0.2, a=1.0)
    p1 = Point(x=float(pair['p1'][0]), y=float(pair['p1'][1]), z=float(pair['p1'][2]))
    p2 = Point(x=float(pair['p2'][0]), y=float(pair['p2'][1]), z=float(pair['p2'][2]))
    tail = position - 0.06 * approach  
    markers = [
        Marker(action=Marker.DELETEALL),
        Marker(header=header, ns='contacts', id=0, type=Marker.LINE_LIST, action=Marker.ADD,
              scale=Vector3(x=0.004), color=yellow, points=[p1, p2]),
        Marker(header=header, ns='contacts', id=1, type=Marker.SPHERE_LIST, action=Marker.ADD,
              scale=Vector3(x=0.014, y=0.014, z=0.014), color=yellow, points=[p1, p2]),
        Marker(header=header, ns='approach', id=2, type=Marker.ARROW, action=Marker.ADD,
              scale=Vector3(x=0.006, y=0.012, z=0.012), color=green,
              points=[Point(x=float(tail[0]), y=float(tail[1]), z=float(tail[2])),
                      Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))]),
    ]
    return MarkerArray(markers=markers)


class PointCloudGrasping(Node):
    def __init__(self):
        super().__init__('pointcloud_processing')
        self.declare_parameter('point_cloud_topic', '/multiview/merged')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('processing_period_sec', 0.5)
        # workspace crop box
        self.declare_parameter('workspace', workspace_from_views_yaml())   
        # table-plane RANSAC
        self.declare_parameter('plane_distance', 0.008)     
        self.declare_parameter('plane_iterations', 1000)
        self.declare_parameter('max_table_tilt_deg', 20.0) 
        self.declare_parameter('plane_attempts', 3)         
        # pulling the objects off the table
        self.declare_parameter('min_height', 0.03)         
        # clustering
        self.declare_parameter('cluster_tolerance', 0.01)  
        self.declare_parameter('cluster_min_size', 80)
        self.declare_parameter('cluster_max_size', 200000)
        # normals
        self.declare_parameter('normal_radius', 0.012)     
        # gripper / friction-cone grasp search
        self.declare_parameter('friction_coefficient', 0.5)  
        self.declare_parameter('min_width', 0.015)            
        self.declare_parameter('max_width', 0.08)          
        self.declare_parameter('close_axis', 'y')             
        # RViz markers
        self.declare_parameter('normal_marker_length', 0.02)  
        self.declare_parameter('normal_marker_stride', 5)     

        self.target_frame = self.get_parameter('target_frame').value
        period = float(self.get_parameter('processing_period_sec').value)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError('processing_period_sec must be finite and positive')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_cloud = None
        self.subscription = self.create_subscription(
            PointCloud2, self.get_parameter('point_cloud_topic').value,
            self.cloud_callback, RELIABLE_QOS,
        )
        self.cloud_publisher = self.create_publisher(PointCloud2, '~/processed_cloud', 1)
        self.cluster_publisher = self.create_publisher(PointCloud2, '~/clusters', 1)
        self.grasp_publisher = self.create_publisher(PoseStamped, '~/grasp_pose', 1)
        self.normals_publisher = self.create_publisher(MarkerArray, '~/normals', 1)
        self.grasp_markers_publisher = self.create_publisher(MarkerArray, '~/grasp_markers', 1)
        self.timer = self.create_timer(period, self.process_latest_cloud)
        self.get_logger().info(f"Listening on {self.get_parameter('point_cloud_topic').value}; "
                               'this node never moves the robot.')

    def cloud_callback(self, msg):
        self.latest_cloud = msg

    def process_latest_cloud(self):
        msg = self.latest_cloud
        if msg is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
            )
        except TransformException as exc:
            self.get_logger().warning(f'Waiting for cloud TF: {exc}', throttle_duration_sec=5.0)
            return
        self.latest_cloud = None
        cloud = cloud_from_msg(msg)
        if len(cloud) == 0:
            return
        cloud = transform_cloud(cloud, transform)

        processed = self.process_cloud(cloud)            
        if processed is None or len(processed) == 0:
            return
        header = Header(stamp=msg.header.stamp, frame_id=self.target_frame)
        self.cloud_publisher.publish(point_cloud2.create_cloud_xyz32(header, processed.xyz))

        clusters = cluster_objects(
            processed,
            tolerance=self.get_parameter('cluster_tolerance').value,
            min_size=self.get_parameter('cluster_min_size').value,
            max_size=self.get_parameter('cluster_max_size').value,
        )
        if clusters:
            points = np.concatenate([c.xyz for c in clusters])
            colors = np.concatenate([np.tile(COLORS[k % len(COLORS)], (len(c), 1))
                                     for k, c in enumerate(clusters)])
            self.cluster_publisher.publish(numpy_to_msg(points, colors, self.target_frame, header.stamp))

        pose = self.estimate_grasp(clusters, header)
        if pose is not None:
            self.grasp_publisher.publish(PoseStamped(header=header, pose=pose))

    def process_cloud(self, cloud):
        log = self.get_logger().info
        n_in = len(cloud)

        # crop to the workspace box
        cloud = crop_workspace(cloud, self.get_parameter('workspace').value)
        if len(cloud) < 100:
            log(f"Only {len(cloud)} points inside the workspace {self.get_parameter('workspace').value}")
            return None
        n_workspace = len(cloud)

        # find the table
        plane, table_indices = segment_table(
            cloud,
            distance_threshold=self.get_parameter('plane_distance').value,
            max_iterations=self.get_parameter('plane_iterations').value,
            max_tilt_deg=self.get_parameter('max_table_tilt_deg').value,
            attempts=self.get_parameter('plane_attempts').value,
            log=log,
        )
        if plane is None:
            log('No horizontal table plane found')
            return None

        # whatever's left standing above it
        z_max = self.get_parameter('workspace').value[5]
        objects = segment_objects(cloud, plane, table_indices,
                                  min_height=self.get_parameter('min_height').value, z_max=z_max)
        log(f'{n_in} points -> {n_workspace} in workspace -> table {len(table_indices)} pts '
            f'at z = {-plane[3] / plane[2]:.3f} m -> {len(objects)} object points')
        return objects

    def estimate_grasp(self, clusters, header) -> Pose | None:
        log = self.get_logger().info
        if not clusters:
            log('No object clusters found')
            self.normals_publisher.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
            self.grasp_markers_publisher.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
            return None
        mu = self.get_parameter('friction_coefficient').value
        min_width = self.get_parameter('min_width').value
        max_width = self.get_parameter('max_width').value
        close_axis = self.get_parameter('close_axis').value
        clusters_data = []
        grasp_pose, grasp_pair, grasp_position, grasp_approach = None, None, None, None
        for k, cluster in enumerate(clusters):
            centroid, cluster, normals = centroid_and_normals(
                cluster, normal_radius=self.get_parameter('normal_radius').value)
            pair, feasible = find_grasp_pair(cluster, normals, mu, min_width, max_width)
            if pair is None:
                log(f'object {k}: {len(cluster)} pts, centroid {np.round(centroid, 3)}: '
                    f'NO feasible contact pair (mu={mu}, {len(normals)} normals, '
                    f'limit {np.degrees(np.arctan(mu)):.1f} deg)')
            else:
                log(f'object {k}: {len(cluster)} pts, centroid {np.round(centroid, 3)}: {feasible} feasible pairs\n'
                    f'    contacts {np.round(pair["p1"], 3)} / {np.round(pair["p2"], 3)}, '
                    f'width {pair["width"] * 1000:.1f} mm\n'
                    f'    normal-to-line angles {pair["angles"][0]:.1f} / {pair["angles"][1]:.1f} deg '
                    f'(limit {np.degrees(np.arctan(mu)):.1f})')
                if grasp_pose is None:                                   
                    position = (pair['p1'] + pair['p2']) / 2.0
                    approach = approach_direction(pair['close_dir'], pair['close_dir'], position)
                    qx, qy, qz, qw = matrix_to_quaternion(grasp_rotation(approach, pair['close_dir'], close_axis))
                    grasp_pose = Pose(position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                                      orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)))
                    grasp_pair, grasp_position, grasp_approach = pair, position, approach
                    log(f'    approach {np.round(approach, 2)}, quaternion (x,y,z,w) {np.round([qx, qy, qz, qw], 3)}')
            clusters_data.append(dict(centroid=centroid, cluster=cluster, normals=normals, pair=pair))
        self.grasp_markers_publisher.publish(
            MarkerArray(markers=[Marker(action=Marker.DELETEALL)]) if grasp_pair is None else
            grasp_marker_array(grasp_pair, grasp_position, grasp_approach, self.target_frame, header.stamp))
        markers = normals_marker_array(
            clusters_data, self.target_frame, header.stamp,
            length=self.get_parameter('normal_marker_length').value,
            stride=self.get_parameter('normal_marker_stride').value,
        )
        self.normals_publisher.publish(markers)
        return grasp_pose


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudGrasping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()