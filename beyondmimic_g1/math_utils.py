from __future__ import annotations

import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1.0e-8:
        raise ValueError("zero-length quaternion")
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quat(q1)
    w2, x2, y2, z2 = normalize_quat(q2)
    return normalize_quat(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = np.trace(matrix)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q = np.array([
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            ])
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q = np.array([
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            ])
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q = np.array([
                (matrix[1, 0] - matrix[0, 1]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
            ])
    return normalize_quat(q)


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)


def rotate_inverse(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q).T @ np.asarray(vector, dtype=np.float64).reshape(3)


def relative_orientation_6d(
    robot_anchor_quat_w: np.ndarray,
    motion_anchor_quat_w: np.ndarray,
    motion_to_world_yaw: np.ndarray,
) -> np.ndarray:
    aligned_motion = quat_multiply(motion_to_world_yaw, motion_anchor_quat_w)
    relative = quat_multiply(quat_conjugate(robot_anchor_quat_w), aligned_motion)
    return quat_to_matrix(relative)[:, :2].reshape(6).astype(np.float32)


def initial_motion_to_world_yaw(
    robot_anchor_quat_w: np.ndarray,
    motion_anchor_quat_w: np.ndarray,
) -> np.ndarray:
    alignment = quat_to_matrix(yaw_quat(robot_anchor_quat_w)) @ quat_to_matrix(yaw_quat(motion_anchor_quat_w)).T
    return matrix_to_quat(alignment)


def tilt_angle(q: np.ndarray) -> float:
    body_up_w = quat_to_matrix(q)[:, 2]
    return float(np.arccos(np.clip(body_up_w[2], -1.0, 1.0)))
