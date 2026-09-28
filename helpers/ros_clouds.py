# Conversions
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


# PointCloud2 to (points (N,3) float32, colors (N,3) uint8 or None)
def msg_to_numpy(msg):
    by_name = {f.name: f for f in msg.fields}
    use_rgb = 'rgb' in by_name and by_name['rgb'].datatype == PointField.FLOAT32
    names = ('x', 'y', 'z', 'rgb') if use_rgb else ('x', 'y', 'z')
    data = point_cloud2.read_points_numpy(msg, field_names=names, skip_nans=False).reshape(-1, len(names))
    xyz = data[:, :3]
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 1e-6)   # no NaNs, no (0,0,0)
    points = np.ascontiguousarray(xyz[valid], dtype=np.float32)
    if not use_rgb:
        return points, None
    packed = np.ascontiguousarray(data[valid, 3], dtype=np.float32).view(np.uint32)
    colors = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1).astype(np.uint8)
    return points, colors


#  x, y, z + rgb
def numpy_to_msg(points, colors, frame_id, stamp):
    header = Header(frame_id=frame_id, stamp=stamp)
    if colors is None:
        return point_cloud2.create_cloud_xyz32(header, points.astype(np.float32))
    cloud = np.zeros(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
    cloud['x'], cloud['y'], cloud['z'] = points[:, 0], points[:, 1], points[:, 2]
    c = colors.astype(np.uint32)
    cloud['rgb'] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate(('x', 'y', 'z', 'rgb'))]
    return PointCloud2(header=header, height=1, width=len(points), fields=fields, is_bigendian=False,
                       point_step=16, row_step=16 * len(points), data=cloud.tobytes(), is_dense=True)
