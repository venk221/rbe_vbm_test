# Point-to-plane ICP in NumPy 
import numpy as np
from scipy.spatial import cKDTree

from .geometry import apply_transform


# normal at each point 
def estimate_normals(points, k=20):
    _, idx = cKDTree(points).query(points, k=k)
    neighbours = points[idx]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


#axis-angle vector rotation matrix or Rodrigues fromula)
def rotvec_to_matrix(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def alignment_error(source, target, max_distance):
    d, _ = cKDTree(target).query(source, distance_upper_bound=max_distance)
    inliers = np.isfinite(d)
    rmse = float(np.sqrt(np.mean(d[inliers] ** 2))) if inliers.any() else float('inf')
    return float(inliers.mean()), rmse


# finds T (4x4) 
def icp_point_to_plane(source, target, max_distance=0.01, iterations=40, tolerance=1e-7):
    center = target.mean(axis=0)           
    src = (source - center).astype(np.float64)
    tgt = (target - center).astype(np.float64)
    normals = estimate_normals(tgt)
    tree = cKDTree(tgt)

    T_local = np.eye(4)
    for _ in range(iterations):
        d, j = tree.query(src, distance_upper_bound=max_distance)
        matched = np.isfinite(d)
        if matched.sum() < 100:
            break
        p, q, n = src[matched], tgt[j[matched]], normals[j[matched]]
        residual = np.einsum('ij,ij->i', p - q, n)
        J = np.hstack([np.cross(p, n), n])           
        dx = np.linalg.lstsq(J, -residual, rcond=None)[0]
        dT = np.eye(4)
        dT[:3, :3] = rotvec_to_matrix(dx[:3])
        dT[:3, 3] = dx[3:]
        src = src @ dT[:3, :3].T + dT[:3, 3]
        T_local = dT @ T_local
        if np.linalg.norm(dx) < tolerance:
            break

    shift, unshift = np.eye(4), np.eye(4)
    shift[:3, 3], unshift[:3, 3] = -center, center
    T = unshift @ T_local @ shift                   
    fitness, rmse = alignment_error(apply_transform(source, T), target, max_distance)
    return T, fitness, rmse
