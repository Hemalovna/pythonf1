"""
formulas3d.py

A compact collection of 3D formulas in Python (numpy) with examples and a small matplotlib demo.

Dependencies:
    pip install numpy matplotlib

Usage:
    python formulas3d.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------
# Basic vector utilities
# ---------------------------
def norm(v):
    """Return Euclidean norm of vector v."""
    v = np.asarray(v, dtype=float)
    return np.linalg.norm(v)


def normalize(v):
    """Return unit vector of v. If zero vector, returns same shape zeros."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return np.zeros_like(v)
    return v / n


def dot(a, b):
    """Dot product of a and b."""
    return float(np.dot(a, b))


def cross(a, b):
    """Cross product a x b."""
    return np.cross(a, b)


def distance(a, b):
    """Euclidean distance between points a and b."""
    return norm(np.asarray(a) - np.asarray(b))


def angle_between(a, b):
    """Angle in radians between vectors a and b (0..pi)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cosv = np.dot(a, b) / (na * nb)
    # clamp numeric errors
    cosv = max(-1.0, min(1.0, cosv))
    return math.acos(cosv)


def projection(a, b):
    """Project vector a onto vector b (returns vector)."""
    b = np.asarray(b, dtype=float)
    bb = np.dot(b, b)
    if bb == 0:
        return np.zeros_like(a)
    return (np.dot(a, b) / bb) * b


def reflect(v, normal):
    """Reflect vector v across plane with unit normal `normal`. Assumes normal may not be unit; it will be normalized."""
    n = normalize(normal)
    return v - 2 * dot(v, n) * n


# ---------------------------
# Rotations: axis-angle, Euler, rotation matrix
# ---------------------------
def rotation_matrix_axis_angle(axis, angle_rad):
    """
    Rodrigues' rotation formula: rotation matrix for rotating by angle around axis.
    axis: 3-vector (doesn't have to be normalized)
    angle_rad: angle in radians
    Returns 3x3 numpy array.
    """
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    R = np.array([
        [c + x*x*C,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  c + z*z*C]
    ], dtype=float)
    return R


def euler_to_rot_matrix(roll, pitch, yaw, order="xyz"):
    """
    Convert Euler angles (radians) to rotation matrix.
    roll -> rotation about x, pitch -> y, yaw -> z by default (order 'xyz').
    order: permutation of 'x','y','z' specifying rotation order.
    """
    Rx = lambda a: rotation_matrix_axis_angle([1, 0, 0], a)
    Ry = lambda a: rotation_matrix_axis_angle([0, 1, 0], a)
    Rz = lambda a: rotation_matrix_axis_angle([0, 0, 1], a)
    mapping = {'x': Rx, 'y': Ry, 'z': Rz}
    R = np.eye(3)
    for axis, angle in zip(order, (roll, pitch, yaw)):
        R = R @ mapping[axis](angle)
    return R


# ---------------------------
# Quaternions (w, x, y, z)
# ---------------------------
def quaternion_from_axis_angle(axis, angle_rad):
    """Return quaternion (w, x, y, z) from axis-angle."""
    axis = normalize(axis)
    s = math.sin(angle_rad / 2.0)
    w = math.cos(angle_rad / 2.0)
    x, y, z = axis * s
    return np.array([w, x, y, z], dtype=float)


def quaternion_mul(q1, q2):
    """Multiply two quaternions q1 * q2. Inputs (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)


def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quaternion_rotate_vector(q, v):
    """Rotate vector v by quaternion q. q=(w,x,y,z)"""
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    q_v = np.concatenate(([0.0], v))
    q_inv = quaternion_conjugate(q) / (np.dot(q, q))
    res = quaternion_mul(quaternion_mul(q, q_v), q_inv)
    return res[1:4]


# ---------------------------
# Planes & intersections
# ---------------------------
def plane_from_points(p0, p1, p2):
    """
    Return plane normal (unit) and d such that plane equation is n.x + d = 0
    given three non-collinear points p0, p1, p2.
    """
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    n = normalize(np.cross(p1 - p0, p2 - p0))
    d = -dot(n, p0)
    return n, d


def line_plane_intersection(line_point, line_dir, plane_normal, plane_d, eps=1e-8):
    """
    Intersect a line (point + t*dir) with plane n.x + d = 0.
    Returns (hit, point, t). If parallel returns (False, None, None).
    """
    line_dir = np.asarray(line_dir, dtype=float)
    denom = dot(plane_normal, line_dir)
    if abs(denom) < eps:
        return False, None, None
    t = -(dot(plane_normal, line_point) + plane_d) / denom
    pt = line_point + t * line_dir
    return True, pt, t


# ---------------------------
# Affine transforms and projections
# ---------------------------
def make_transform_matrix(translation=(0, 0, 0), rotation_matrix=None, scale=(1, 1, 1)):
    """Return a 4x4 homogeneous transform matrix."""
    T = np.eye(4, dtype=float)
    sx, sy, sz = scale
    S = np.diag([sx, sy, sz, 1.0])
    T[:3, 3] = translation
    if rotation_matrix is None:
        R = np.eye(3)
    else:
        R = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    M = T.copy()
    M[:3, :3] = R @ np.diag([sx, sy, sz])
    return M


def transform_point(M, p):
    """Apply 4x4 transform M to 3D point p (returns 3-vector)."""
    v = np.asarray([p[0], p[1], p[2], 1.0], dtype=float)
    r = M @ v
    return (r[:3] / r[3]) if r[3] != 0 else r[:3]


def perspective_project_point(p, fov_y_deg=60, aspect=1.0, near=0.1, far=1000.0):
    """
    Very simple perspective projection of point p (camera at origin looking +Z).
    Returns normalized device coordinates (x_ndc, y_ndc) and depth z.
    This is illustrative; for real rendering use proper camera matrices.
    """
    fov_y = math.radians(fov_y_deg)
    f = 1.0 / math.tan(fov_y / 2.0)
    x, y, z = p
    if z <= 0:
        # behind camera; return None or projected mirror (we choose None)
        return None, None, z
    x_ndc = (x * f / aspect) / z
    y_ndc = (y * f) / z
    return x_ndc, y_ndc, z


# ---------------------------
# Geometry helpers
# ---------------------------
def closest_point_on_segment(a, b, p):
    """Return closest point to p on segment ab."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    ab = b - a
    t = dot(p - a, ab) / dot(ab, ab) if dot(ab, ab) != 0 else 0.0
    t = max(0.0, min(1.0, t))
    return a + t * ab


# ---------------------------
# Demo / Examples
# ---------------------------
def demo():
    print("=== 3D formulas demo ===")
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([-1.0, 0.5, 2.0])
    print("v1:", v1)
    print("v2:", v2)
    print("norm(v1):", norm(v1))
    print("normalize(v2):", normalize(v2))
    print("dot:", dot(v1, v2))
    print("cross:", cross(v1, v2))
    print("angle (deg):", math.degrees(angle_between(v1, v2)))

    # Rotate v1 90 degrees around Y
    R = rotation_matrix_axis_angle([0, 1, 0], math.pi / 2)
    print("Rotation matrix (Y 90 deg):\n", R)
    print("R @ v1:", R @ v1)

    # Quaternion rotation same rotation:
    q = quaternion_from_axis_angle([0, 1, 0], math.pi / 2)
    print("Quaternion:", q)
    print("Quaternion rotate v1:", quaternion_rotate_vector(q, v1))

    # Plane from 3 points and line intersection
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    n, d = plane_from_points(p0, p1, p2)
    print("Plane normal, d:", n, d)
    line_pt = np.array([0.2, 0.2, 1.0])
    line_dir = np.array([0.0, 0.0, -1.0])
    hit, pt, t = line_plane_intersection(line_pt, line_dir, n, d)
    print("Line-plane hit:", hit, "point:", pt, "t:", t)

    # transform and projection demo: rotate a cube and plot
    cube = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=float)

    # build rotation about axis [1,1,0]
    axis = np.array([1.0, 1.0, 0.0])
    Rcube = rotation_matrix_axis_angle(axis, math.radians(35))
    M = make_transform_matrix(translation=(0.0, 0.0, 6.0), rotation_matrix=Rcube, scale=(0.7, 0.7, 0.7))
    cube_t = np.array([transform_point(M, p) for p in cube])

    # plotting
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Rotated cube (demo)")
    # plot cube vertices
    ax.scatter(cube_t[:,0], cube_t[:,1], cube_t[:,2], s=40)
    # draw edges by index pairs
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        xs = [cube_t[i,0], cube_t[j,0]]
        ys = [cube_t[i,1], cube_t[j,1]]
        zs = [cube_t[i,2], cube_t[j,2]]
        ax.plot(xs, ys, zs)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.auto_scale_xyz([-3,3], [-3,3], [0,8])
    plt.show()


if __name__ == "__main__":
    demo()
