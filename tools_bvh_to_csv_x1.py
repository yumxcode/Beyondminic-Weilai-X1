#!/usr/bin/env python3
"""Retarget a LAFAN1 BVH file to a BeyondMimic-compatible Xyber X1 CSV.

Headless counterpart of GMR/scripts/bvh_to_robot.py: no MuJoCo viewer, keeps
all frames, writes the 36-column CSV consumed by csv_to_npz.py --robot x1.

Column layout (must match X1_JOINT_NAMES in csv_to_npz.py):

    root_pos_xyz, root_quat_xyzw, 29 X1 joint positions

Run inside the GMR Python environment:
    python tools_bvh_to_csv_x1.py --bvh_file downloads/lafan1/dance1_subject2.bvh \
        --output_file whole_body_tracking/motions/csv/dance1_subject2_x1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DEFAULT_GMR_ROOT = REPO / "GMR"

# canonical X1 CSV column order (mirrors csv_to_npz.py X1_JOINT_NAMES)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget LAFAN1 BVH to a BeyondMimic X1 CSV.")
    parser.add_argument("--bvh_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--gmr_root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--target_fps", type=int, default=30, help="LAFAN1 BVH is 30 FPS.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--use_velocity_limit", action="store_true")
    parser.add_argument("--solver", default="proxqp")
    parser.add_argument("--init_margin", type=float, default=0.15,
                        help="Initial pose fraction into each joint range (avoids limit-edge IK failures).")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Centered moving-average window over joint+root trajectories (0 disables). "
                        "5 frames @30fps ~ 6 Hz cutoff: keeps dance content, removes IK jitter and "
                        "bounds joint velocities within URDF limits.")
    return parser.parse_args()


def make_quaternions_continuous(quats_xyzw: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats_xyzw, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Retargeting produced a zero-length root quaternion")
    quats_xyzw = quats_xyzw / norms
    for frame in range(1, len(quats_xyzw)):
        if np.dot(quats_xyzw[frame - 1], quats_xyzw[frame]) < 0.0:
            quats_xyzw[frame] *= -1.0
    return quats_xyzw


def smooth_motion(motion: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over all columns; quaternions renormalized."""
    if window <= 1:
        return motion
    kernel = np.ones(window) / window
    out = motion.copy()
    # reflect-pad edges so the motion length is preserved
    pad = window // 2
    padded = np.concatenate([motion[: pad][::-1], motion, motion[-pad:][::-1]], axis=0)
    for c in range(motion.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="same")[pad:pad + len(motion)]
    # renormalize root quaternion columns (3:7, xyzw)
    norms = np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    out[:, 3:7] /= norms
    return out


def main() -> None:
    args = parse_args()
    if not args.bvh_file.is_file():
        raise FileNotFoundError(args.bvh_file)
    gmr_root = args.gmr_root.resolve()
    if not (gmr_root / "general_motion_retargeting").is_dir():
        raise FileNotFoundError(f"GMR package not found under {gmr_root}")
    sys.path.insert(0, str(gmr_root))

    import mujoco  # noqa: F401  (needed by GMR)
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_height = load_bvh_file(str(args.bvh_file), format="lafan1")
    end_frame = len(frames) if args.end_frame is None else args.end_frame
    if args.start_frame >= end_frame:
        raise ValueError("empty frame range")

    retargeter = GeneralMotionRetargeting(
        actual_human_height=float(human_height),
        src_human="bvh_lafan1",
        tgt_robot="xyber_x1",
        solver=args.solver,
        verbose=False,
        use_velocity_limit=args.use_velocity_limit,
    )

    import mujoco as mj

    model = retargeter.model
    qpos_adrs, limits = [], []
    for name in X1_JOINT_NAMES:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint {name} missing from X1 mocap model")
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        limits.append(model.jnt_range[jid].copy())
    qpos_adrs = np.array(qpos_adrs)
    lo = np.array([l[0] for l in limits])
    up = np.array([l[1] for l in limits])

    # start strictly inside the (margin-widened) limits
    init = retargeter.configuration.data.qpos.copy()
    for adr, l, u in zip(qpos_adrs, lo, up):
        init[adr] = l + args.init_margin * (u - l)
    retargeter.configuration.update(init)
    retargeter.max_iter = 30  # more IK iterations for dynamic frames (default 10)

    # NOTE: a mink VelocityLimit makes the IK QP transiently infeasible on fast
    # dance frames (NoSolutionFound); output velocity bounds are instead
    # enforced by the post-hoc smoothing pass (--smooth_window).

    qpos_frames = []
    failed_frames = []
    for index in range(args.start_frame, end_frame):
        frame = frames[index]
        try:
            qpos = np.asarray(retargeter.retarget(frame), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — freeze-frame fallback
            print(f"[warn] frame {index}: retarget failed ({type(exc).__name__}), freezing pose")
            failed_frames.append(index)
            qpos = qpos_frames[-1].copy() if qpos_frames else None
            if qpos is None:
                continue
        if qpos.shape != (36,):
            raise RuntimeError(f"expected qpos (36,), got {qpos.shape} at frame {index}")
        # clamp joints back to the *real* URDF limits (mocap xml has a small margin)
        qpos[qpos_adrs] = np.clip(qpos[qpos_adrs], lo + 0.01, up - 0.01)
        qpos_frames.append(qpos)
        if (index - args.start_frame) % 300 == 0:
            print(f"frame {index}/{end_frame}")

    motion = np.stack(qpos_frames).astype(np.float32)
    if not np.all(np.isfinite(motion)):
        bad = np.argwhere(~np.isfinite(motion))[0]
        raise RuntimeError(f"non-finite value at frame {bad[0]}, col {bad[1]}")

    if args.smooth_window > 1:
        motion = smooth_motion(motion.astype(np.float64), args.smooth_window).astype(np.float32)
        # smoothing can push joints marginally past limits again — re-clamp
        motion[:, qpos_adrs - 0] = np.clip(motion[:, qpos_adrs - 0], lo + 0.01 - 1e-6, up - 0.01 + 1e-6)
        print(f"[smooth] applied centered moving average, window={args.smooth_window}")

    # MuJoCo root quat is wxyz; CSV loader expects xyzw
    motion[:, 3:7] = motion[:, [4, 5, 6, 3]]
    motion[:, 3:7] = make_quaternions_continuous(motion[:, 3:7])

    out = args.output_file.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, motion, delimiter=",", fmt="%.8f")
    duration = (len(motion) - 1) / float(args.target_fps) if len(motion) > 1 else 0.0
    print(f"Saved {len(motion)} frames ({duration:.2f}s at {args.target_fps} FPS) to {out}; frozen frames: {len(failed_frames)}")
    print("CSV layout: root xyz + quat xyzw + 29 X1 joints")


if __name__ == "__main__":
    main()
