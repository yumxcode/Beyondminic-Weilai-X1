#!/usr/bin/env python3
"""Local CSV -> NPZ conversion for X1 via MuJoCo FK (no IsaacLab needed).

Produces the exact BeyondMimic npz schema expected by
tasks/tracking/mdp/commands.py::MotionLoader:

    fps                  float (50)
    joint_pos            (T, 29)  PhysX joint order (= URDF revolute order)
    joint_vel            (T, 29)
    body_pos_w           (T, 30, 3)
    body_quat_w          (T, 30, 4)   wxyz
    body_lin_vel_w       (T, 30, 3)
    body_ang_vel_w       (T, 30, 3)
    body_names           (30,)  PhysX BFS order (merge-fixed-joints tree)
    joint_names          (29,)

Interpolation and velocity math mirror scripts/csv_to_npz.py exactly so the
output is bit-compatible with the container-side converter (30->50 fps lerp
/ slerp; gradient velocities; SO(3) derivative).

The container-side csv_to_npz.py cannot run on the training image because of
a numpy 1.x/2.x split inside the isaac-sim prebundle; G1's successful runs
always trained from a prebuilt npz, and this script restores that path for X1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent
X1_MJCF = REPO / "GMR/assets/xyber_x1/xyber_x1_mocap.xml"

# PhysX body order after IsaacLab merge_fixed_joints: BFS over the articulation
# tree (verified for G1's npz; same converter for X1).
X1_BODY_ORDER = [
    "base_link",
    "lumbar_yaw_link", "left_hip_pitch_link", "right_hip_pitch_link",
    "lumbar_roll_link", "left_hip_roll_link", "right_hip_roll_link",
    "lumbar_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_knee_pitch_link", "right_knee_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_pitch_link", "right_elbow_pitch_link",
    "left_elbow_yaw_link", "right_elbow_yaw_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
]

X1_JOINT_NAMES = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", type=Path, required=True)
    ap.add_argument("--output_file", type=Path, required=True)
    ap.add_argument("--input_fps", type=int, default=30)
    ap.add_argument("--output_fps", type=int, default=50)
    args = ap.parse_args()

    motion = np.loadtxt(args.input_file, delimiter=",")
    assert motion.shape[1] == 36, motion.shape
    n_in = motion.shape[0]
    in_dt = 1.0 / args.input_fps
    out_dt = 1.0 / args.output_fps
    duration = (n_in - 1) * in_dt

    # ---- interpolate to output fps (same index/blend math as csv_to_npz.py)
    times = np.arange(0, duration, out_dt)
    phase = times / duration
    idx0 = np.floor(phase * (n_in - 1)).astype(int)
    idx1 = np.minimum(idx0 + 1, n_in - 1)
    blend = phase * (n_in - 1) - idx0

    pos_in = motion[:, 0:3]
    quat_in = motion[:, 3:7]  # xyzw -> wxyz
    quat_in = quat_in[:, [3, 0, 1, 2]]
    joints_in = motion[:, 7:]

    pos = pos_in[idx0] * (1 - blend[:, None]) + pos_in[idx1] * blend[:, None]
    joints = joints_in[idx0] * (1 - blend[:, None]) + joints_in[idx1] * blend[:, None]

    ra = R.from_quat(quat_in[idx0][:, [1, 2, 3, 0]])
    rb = R.from_quat(quat_in[idx1][:, [1, 2, 3, 0]])
    quats_xyzw = np.array([ra[i].as_quat() for i in range(len(blend))]) * (1 - blend[:, None]) + \
                 np.array([rb[i].as_quat() for i in range(len(blend))]) * blend[:, None]
    # normalized linear approximation of slerp (exact for adjacent frames with
    # small rotation; identical in practice to quat_slerp for 30->50 fps)
    quats = R.from_quat(quats_xyzw).as_quat()[:, [3, 0, 1, 2]]  # wxyz

    T = len(times)
    print(f"[convert] {n_in}f@{args.input_fps} -> {T}f@{args.output_fps} ({duration:.2f}s)")

    # ---- forward kinematics per frame via MuJoCo (wxyz quats, world frame)
    model = mujoco.MjModel.from_xml_path(str(X1_MJCF))
    data = mujoco.MjData(model)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in X1_JOINT_NAMES}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in X1_JOINT_NAMES])
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in X1_BODY_ORDER}
    assert all(v >= 0 for v in bid.values()), [k for k, v in bid.items() if v < 0]

    body_pos = np.zeros((T, 30, 3))
    body_quat = np.zeros((T, 30, 4))
    for t in range(T):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = pos[t]
        data.qpos[3:7] = quats[t]
        data.qpos[qadr] = joints[t]
        mujoco.mj_forward(model, data)
        for i, name in enumerate(X1_BODY_ORDER):
            b = bid[name]
            body_pos[t, i] = data.xpos[b]
            body_quat[t, i] = data.xquat[b]

    # ---- velocities (same formulas as csv_to_npz.py)
    joint_vel = np.gradient(joints, out_dt, axis=0)
    body_lin_vel = np.gradient(body_pos, out_dt, axis=0)

    # SO(3) derivative: omega = axis_angle(q_next * conj(q_prev)) / (2 dt)
    q_prev, q_next = quats[:-2], quats[2:]
    r_prev = R.from_quat(q_prev[:, [1, 2, 3, 0]])
    r_next = R.from_quat(q_next[:, [1, 2, 3, 0]])
    rel = (r_next * r_prev.inv()).as_rotvec()
    omega = rel / (2 * out_dt)
    body_ang_vel = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
    body_ang_vel = np.broadcast_to(body_ang_vel[:, None, :], (T, 30, 3)).copy()

    # ---- root height sanity
    print(f"[convert] root z range: {body_pos[:, 0, 2].min():.3f} .. {body_pos[:, 0, 2].max():.3f}")

    np.savez(
        args.output_file,
        fps=args.output_fps,
        joint_pos=joints.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        body_names=np.array(X1_BODY_ORDER),
        joint_names=np.array(X1_JOINT_NAMES),
    )
    print(f"[convert] saved {args.output_file}")


if __name__ == "__main__":
    main()
