#!/usr/bin/env python3
"""Retarget a GVHMR result to a BeyondMimic-compatible Unitree G1 CSV.

Run this script in the GMR Python environment.  It deliberately does not
create a MuJoCo viewer, which makes it suitable for a headless machine.

The output columns are::

    root_pos_xyz, root_quat_xyzw, 29 G1 joint positions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GMR_ROOT = REPO_ROOT / "GMR"

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget GVHMR SMPL-X output to a BeyondMimic G1 CSV."
    )
    parser.add_argument(
        "--gvhmr_pred_file",
        type=Path,
        required=True,
        help="GVHMR hmr4d_results.pt file.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--gmr_root",
        type=Path,
        default=DEFAULT_GMR_ROOT,
        help=f"GMR repository root (default: {DEFAULT_GMR_ROOT}).",
    )
    parser.add_argument(
        "--body_model_dir",
        type=Path,
        default=None,
        help="Directory containing smplx/SMPLX_NEUTRAL.pkl. Defaults to GMR/assets/body_models.",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Retargeting frame rate. GVHMR output is normally 30 FPS.",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="First frame to keep, inclusive.")
    parser.add_argument("--end_frame", type=int, default=None, help="Last frame to keep, exclusive.")
    parser.add_argument(
        "--use_velocity_limit",
        action="store_true",
        help="Enable GMR's per-joint IK velocity limit.",
    )
    parser.add_argument(
        "--solver",
        default="proxqp",
        help="QP solver used by GMR (default: proxqp, installed by GMR's setup.py).",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> Path:
    pred_file = args.gvhmr_pred_file.expanduser().resolve()
    gmr_root = args.gmr_root.expanduser().resolve()
    body_model_dir = (
        args.body_model_dir.expanduser().resolve()
        if args.body_model_dir is not None
        else gmr_root / "assets" / "body_models"
    )

    if not pred_file.is_file():
        raise FileNotFoundError(f"GVHMR result not found: {pred_file}")
    if not (gmr_root / "general_motion_retargeting").is_dir():
        raise FileNotFoundError(f"GMR package not found under: {gmr_root}")
    neutral_model = body_model_dir / "smplx" / "SMPLX_NEUTRAL.pkl"
    if not neutral_model.is_file():
        raise FileNotFoundError(
            "GMR requires the licensed SMPL-X neutral model at "
            f"{neutral_model}"
        )

    args.gvhmr_pred_file = pred_file
    args.gmr_root = gmr_root
    return body_model_dir


def make_quaternions_continuous(quats_xyzw: np.ndarray) -> np.ndarray:
    """Normalize quaternions and remove sign flips between adjacent frames."""
    norms = np.linalg.norm(quats_xyzw, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Retargeting produced a zero-length root quaternion")
    quats_xyzw = quats_xyzw / norms
    for frame in range(1, len(quats_xyzw)):
        if np.dot(quats_xyzw[frame - 1], quats_xyzw[frame]) < 0.0:
            quats_xyzw[frame] *= -1.0
    return quats_xyzw


def main() -> None:
    args = parse_args()
    if args.target_fps <= 0:
        raise ValueError("--target_fps must be positive")
    if args.start_frame < 0:
        raise ValueError("--start_frame must be non-negative")

    body_model_dir = validate_paths(args)
    sys.path.insert(0, str(args.gmr_root))

    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.smpl import (
        get_gvhmr_data_offline_fast,
        load_gvhmr_pred_file,
    )

    smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
        str(args.gvhmr_pred_file), str(body_model_dir)
    )
    frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.target_fps
    )

    end_frame = len(frames) if args.end_frame is None else args.end_frame
    if end_frame > len(frames):
        raise ValueError(f"--end_frame={end_frame} exceeds motion length {len(frames)}")
    if args.start_frame >= end_frame:
        raise ValueError("The selected frame range is empty")

    retargeter = GeneralMotionRetargeting(
        actual_human_height=float(human_height),
        src_human="smplx",
        tgt_robot="unitree_g1",
        solver=args.solver,
        verbose=False,
        use_velocity_limit=args.use_velocity_limit,
    )

    actual_joint_names = tuple(
        name for name in retargeter.robot_dof_names if name not in (None, "root")
    )
    # MuJoCo reports the floating base as six unnamed velocity DoFs. The motor
    # order is the unambiguous order used by qpos[7:].
    actual_motor_names = tuple(retargeter.robot_motor_names)
    if actual_motor_names != G1_JOINT_NAMES:
        raise RuntimeError(
            "GMR G1 motor order does not match BeyondMimic.\n"
            f"Expected: {G1_JOINT_NAMES}\nActual:   {actual_motor_names}\n"
            f"Reported DoFs: {actual_joint_names}"
        )
    import mujoco

    qpos_addresses = tuple(
        int(
            retargeter.model.jnt_qposadr[
                mujoco.mj_name2id(retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
        )
        for name in G1_JOINT_NAMES
    )
    if qpos_addresses != tuple(range(7, 36)):
        raise RuntimeError(
            "GMR G1 qpos layout does not match BeyondMimic: "
            f"joint qpos addresses are {qpos_addresses}"
        )

    qpos_frames = []
    for index in range(args.start_frame, end_frame):
        qpos = np.asarray(retargeter.retarget(frames[index]), dtype=np.float64)
        if qpos.shape != (36,):
            raise RuntimeError(f"Expected G1 qpos shape (36,), got {qpos.shape} at frame {index}")
        qpos_frames.append(qpos)

    motion = np.stack(qpos_frames).astype(np.float32)
    if not np.all(np.isfinite(motion)):
        bad = np.argwhere(~np.isfinite(motion))[0]
        raise RuntimeError(f"Non-finite motion value at output frame {bad[0]}, column {bad[1]}")

    # GMR/MuJoCo qpos stores the root quaternion as wxyz; BeyondMimic's CSV
    # loader expects xyzw and converts it to wxyz internally.
    motion[:, 3:7] = motion[:, [4, 5, 6, 3]]
    motion[:, 3:7] = make_quaternions_continuous(motion[:, 3:7])

    output_file = args.output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_file, motion, delimiter=",", fmt="%.8f")

    duration = (len(motion) - 1) / float(aligned_fps) if len(motion) > 1 else 0.0
    print(f"Saved {len(motion)} frames ({duration:.2f}s at {aligned_fps:.3f} FPS) to {output_file}")
    print("CSV layout verified: root xyz + root quaternion xyzw + 29 G1 joints")


if __name__ == "__main__":
    main()
