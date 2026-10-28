"""Camera extrinsics for G1 head front camera (matches RayCaster / RTX CameraCfg)."""

from __future__ import annotations

import numpy as np

# Same as build_rtx_camera_cfg / build_depth_camera_cfgs in cluster_camera_utils.py
CAM_OFFSET_POS = (0.3, 0.0, 0.5)
CAM_OFFSET_QUAT_WXYZ = (0.0, 0.924, 0.0, 0.383)  # ros convention


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (w, x, y, z)."""
    m = np.asarray(r, dtype=np.float64).reshape(3, 3)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def pose_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_wxyz_to_rot(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return t


def camera_world_pose(head_pos: np.ndarray, head_quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World pose of front camera optical frame (ROS: x forward, y left, z up)."""
    t_head = pose_to_matrix(head_pos, head_quat_wxyz)
    t_offset = pose_to_matrix(np.array(CAM_OFFSET_POS), np.array(CAM_OFFSET_QUAT_WXYZ))
    t_cam = t_head @ t_offset
    return t_cam[:3, 3].copy(), rot_to_quat_wxyz(t_cam[:3, :3])


def pinhole_intrinsics(
    height: int,
    width: int,
    focal_length_mm: float = 24.0,
    horizontal_aperture_mm: float = 20.955,
) -> tuple[float, float, float, float]:
    """Return fx, fy, cx, cy in pixels (Isaac PinholeCameraCfg defaults)."""
    vertical_aperture_mm = horizontal_aperture_mm * height / width
    fx = focal_length_mm / horizontal_aperture_mm * width
    fy = focal_length_mm / vertical_aperture_mm * height
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy
