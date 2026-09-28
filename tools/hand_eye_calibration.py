#!/usr/bin/env python3

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener


MOUNT = 'eye_in_hand'        # 'eye_in_hand' or 'eye_to_hand' 
MARKER_LENGTH = 0.1       
MARKER_ID = 0
ARUCO_DICT = cv2.aruco.DICT_4X4_50   # use cv2.aruco.DICT_ARUCO_ORIGINAL only if the aruco marker is "Original ArUco"

BASE_FRAME = 'base_link'
EE_FRAME = 'end_effector_link'
CAMERA_LINK = 'camera_link'
IMAGE_TOPIC = '/camera/camera/color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'
RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'hand_eye_result.txt')


# ArUco detection
DICTIONARY = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
if hasattr(cv2.aruco, 'ArucoDetector'):
    _detector = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())

    def detect(gray):
        return _detector.detectMarkers(gray)
else:
    _params = cv2.aruco.DetectorParameters_create()

    def detect(gray):
        return cv2.aruco.detectMarkers(gray, DICTIONARY, parameters=_params)

_h = MARKER_LENGTH / 2.0
MARKER_CORNERS = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]], dtype=np.float64)


def transform_to_matrix(tr):
    x, y, z, w = tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return T



def matrix_to_quaternion(R):
    qw = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    qx = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    qy = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    qz = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return qx, qy, qz, qw


class Calibrator(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.image = None
        self.camera_frame = None
        self.K = None
        self.D = None
        self.create_subscription(Image, IMAGE_TOPIC, self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, INFO_TOPIC, self.on_info, qos_profile_sensor_data)

    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def on_image(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.image = img.copy()
        self.camera_frame = msg.header.frame_id   


    def lookup(self, target, source):
        return transform_to_matrix(self.tf_buffer.lookup_transform(target, source, Time()).transform)


def find_marker(image, K, D):
    corners, ids, _ = detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None or MARKER_ID not in ids.flatten():
        return None, None, None, corners, ids
    c = corners[list(ids.flatten()).index(MARKER_ID)].reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(MARKER_CORNERS, c, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, None, corners, ids
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.flatten()
    return T, rvec, tvec, corners, ids


def compute(node, robot_poses, marker_poses):
    n = len(robot_poses)
    if n < 5:
        print(f'Only {n} samples. Collect at least 5 (15-20 recommended).')
        return

    # Eye-in-hand
    robot = robot_poses if MOUNT == 'eye_in_hand' else [np.linalg.inv(T) for T in robot_poses]
    R, t = cv2.calibrateHandEye(
        [T[:3, :3] for T in robot], [T[:3, 3] for T in robot],
        [T[:3, :3] for T in marker_poses], [T[:3, 3] for T in marker_poses],
        method=cv2.CALIB_HAND_EYE_TSAI)
    T_parent_optical = np.eye(4)
    T_parent_optical[:3, :3] = R
    T_parent_optical[:3, 3] = t.flatten()

    marker_positions = np.array([(robot[i] @ T_parent_optical @ marker_poses[i])[:3, 3] for i in range(n)])
    spread_mm = np.linalg.norm(marker_positions - marker_positions.mean(axis=0), axis=1) * 1000
    print(f'\nConsistency over {n} samples: mean {spread_mm.mean():.1f} mm, max {spread_mm.max():.1f} mm')
    print('(good: mean below ~3 mm; above ~10 mm means bad samples, recollect with more rotation)')

    T_camlink_optical = node.lookup(CAMERA_LINK, node.camera_frame)
    T_parent_camlink = T_parent_optical @ np.linalg.inv(T_camlink_optical)

    parent = EE_FRAME if MOUNT == 'eye_in_hand' else BASE_FRAME
    x, y, z = T_parent_camlink[:3, 3]
    qx, qy, qz, qw = matrix_to_quaternion(T_parent_camlink[:3, :3])
    command = (f'ros2 run tf2_ros static_transform_publisher '
               f'--x {x:.5f} --y {y:.5f} --z {z:.5f} '
               f'--qx {qx:.6f} --qy {qy:.6f} --qz {qz:.6f} --qw {qw:.6f} '
               f'--frame-id {parent} --child-frame-id {CAMERA_LINK}')
    print(f'\nResult ({parent} -> {CAMERA_LINK}). Publish it with:\n\n{command}\n')
    with open(RESULT_FILE, 'w') as f:
        f.write(f'# {n} samples, consistency mean {spread_mm.mean():.1f} mm, max {spread_mm.max():.1f} mm\n')
        f.write(command + '\n')
    print(f'Saved to {RESULT_FILE}')


def main():
    rclpy.init()
    node = Calibrator()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print('Click the image window, then press: s = save a sample (arm standing still, marker visible), '
         'c = compute (after 15-20 samples), q = quit')

    robot_poses, marker_poses = [], []
    while rclpy.ok():
        if node.image is None or node.K is None:
            print('Waiting for camera image and camera_info ...')
            time.sleep(1.0)
            continue

        image = node.image.copy()
        T_cam_marker, rvec, tvec, corners, ids = find_marker(image, node.K, node.D)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        if T_cam_marker is not None:
            cv2.drawFrameAxes(image, node.K, node.D, rvec, tvec, MARKER_LENGTH)
            status = f'marker at {np.linalg.norm(T_cam_marker[:3, 3]) * 100:.0f} cm'
        else:
            status = 'marker NOT found'
        cv2.putText(image, f'{status} | samples: {len(robot_poses)} | s=save c=compute q=quit',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('hand-eye calibration', image)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            if T_cam_marker is None:
                print('Marker not visible, sample NOT saved.')
                continue
            try:
                T_base_ee = node.lookup(BASE_FRAME, EE_FRAME)
            except Exception as e:
                print(f'TF lookup {BASE_FRAME} -> {EE_FRAME} failed (is the MoveIt launch running?): {e}')
                continue
            robot_poses.append(T_base_ee)
            marker_poses.append(T_cam_marker)
            print(f'Saved sample {len(robot_poses)} ({status})')
        elif key == ord('c'):
            compute(node, robot_poses, marker_poses)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()