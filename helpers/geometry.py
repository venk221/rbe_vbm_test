# geomtrty functions
import numpy as np


def quat_to_matrix(x, y, z, w):
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n == 0.0:
        raise ValueError('quaternion must be finite and nonzero')
    x, y, z, w = q / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform_to_matrix(transform):
    r, t = transform.rotation, transform.translation
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(r.x, r.y, r.z, r.w)
    T[:3, 3] = (t.x, t.y, t.z)
    return T


def apply_transform(points, T):
    return (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def crop_box(points, colors, box):
    keep = np.ones(len(points), dtype=bool)
    for axis, name in enumerate('xyz'):
        lo, hi = box[name]
        keep &= (points[:, axis] >= lo) & (points[:, axis] <= hi)
    return points[keep], (colors[keep] if colors is not None else None)


# averages every point in a voxel down to one (colors too)
def voxel_downsample(points, voxel, colors=None):
    if len(points) == 0:
        return points, colors
    idx = np.floor(points / voxel).astype(np.int64)
    idx -= idx.min(axis=0)
    dims = idx.max(axis=0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    def mean_of(values):
        return np.stack([np.bincount(inverse, weights=values[:, i]) for i in range(3)], axis=1) / counts[:, None]

    out_points = mean_of(points.astype(np.float64)).astype(np.float32)
    out_colors = None
    if colors is not None and len(colors):
        out_colors = np.clip(np.round(mean_of(colors.astype(np.float64))), 0, 255).astype(np.uint8)
    return out_points, out_colors


# drops the D405's "flying pixel" noise at object edges
def remove_outliers(points, colors=None, k=20, std_ratio=2.0):
    if len(points) <= k:
        return points, colors
    from scipy.spatial import cKDTree
    d, _ = cKDTree(points).query(points, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    keep = mean_d <= mean_d.mean() + std_ratio * mean_d.std()
    return points[keep], (colors[keep] if colors is not None else None)


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def save_cloud(path, points, colors=None, **extra):
    np.savez_compressed(path, points=points,
                        colors=colors if colors is not None else np.zeros((0, 3), np.uint8), **extra)


def load_cloud(path):
    data = dict(np.load(path))
    colors = data['colors'] if len(data['colors']) else None
    return data['points'], colors, data


# ASCII PLY, readable by CloudCompare, MeshLab, Open3D, pcl_viewer
def save_ply(path, points, colors=None):
    with open(path, 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'
                'property float x\nproperty float y\nproperty float z\n')
        if colors is not None:
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        if colors is not None:
            for p, c in zip(points, colors):
                f.write(f'{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n')
        else:
            np.savetxt(f, points, fmt='%.5f')
