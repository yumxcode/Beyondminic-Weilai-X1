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

X1_JOINT_NAMES = (
    "lumbar_yaw_joint",
    "lumbar_roll_joint",
    "lumbar_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

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
        "--robot",
        choices=["g1", "x1"],
        default="g1",
        help="Target robot for the BeyondMimic CSV.",
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

    robot_name = "xyber_x1" if args.robot == "x1" else "unitree_g1"
    joint_names_ref = X1_JOINT_NAMES if args.robot == "x1" else G1_JOINT_NAMES
    retargeter = GeneralMotionRetargeting(
        actual_human_height=float(human_height),
        src_human="smplx",
        tgt_robot=robot_name,
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
    expected_order = list(G1_JOINT_NAMES) if args.robot == "g1" else list(X1_JOINT_NAMES)
    actual_hinge_order = tuple(
        name for name in retargeter.robot_dof_names if name not in (None, "floating_base")
    )
    # GMR MuJoCo models may list DoFs in tree order; remap by name instead of
    # assuming positional equality for X1.
    if set(actual_hinge_order) != set(expected_order):
        raise RuntimeError(
            f"GMR {robot_name} joint set does not match BeyondMimic.\n"
            f"Expected: {expected_order}\nActual:   {actual_hinge_order}"
        )
    import mujoco

    import mujoco as mj

    name_to_qadr = {}
    for jid in range(retargeter.model.njnt):
        jname = mj.mj_id2name(retargeter.model, mj.mjtObj.mjOBJ_JOINT, jid)
        if jname and jname != "floating_base":
            name_to_qadr[jname] = int(retargeter.model.jnt_qposadr[jid])

    # Reorder the per-frame qpos from GMR's internal joint order into the
    # canonical CSV column order for the chosen robot.
    internal_hinge_order = list(actual_hinge_order)
    remap = [name_to_qadr[name] for name in expected_order]

    # X1 needs an interior-start initialization (knee/elbow ranges start at 0)
    if args.robot == "x1":
        init = retargeter.configuration.data.qpos.copy()
        for jname, qadr in name_to_qadr.items():
            jid = mj.mj_name2id(retargeter.model, mj.mjtObj.mjOBJ_JOINT, jname)
            lo, up = retargeter.model.jnt_range[jid]
            if lo < up:
                init[qadr] = lo + 0.15 * (up - lo)
        retargeter.configuration.update(init)

    qpos_frames = []
    for index in range(args.start_frame, end_frame):
        try:
            qpos = np.asarray(retargeter.retarget(frames[index]), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] frame {index}: retarget failed ({type(exc).__name__}), freezing pose")
            qpos_frames.append(qpos_frames[-1].copy())
            continue
        if qpos.shape != (36,):
            raise RuntimeError(f"Expected qpos shape (36,), got {qpos.shape} at frame {index}")
        reordered = qpos.copy()
        reordered[7:] = qpos[remap]
        qpos_frames.append(reordered)

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
    print(f"CSV layout verified: root xyz + root quaternion xyzw + 29 {args.robot.upper()} joints")


if __name__ == "__main__":
    main()
