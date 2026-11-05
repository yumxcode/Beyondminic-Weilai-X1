#!/usr/bin/env python3
"""Retargeting quality gate for F1 motions.

Same six checks and thresholds as tools_check_motion_x1.py (the validated X1
gate), with F1 geometry:

  1. ground_penetration  — F1 foot sole AABB corner spheres (vendor collision
                           STL, from tools_build_f1_mjcf.py audit output)
  2. self_penetration    — sphere-per-link approximations via mj_geomDistance
  3. joint_limits        — F1 URDF limits (asymmetric hip_roll ±20°/90°,
                           ankle ±0.524, knee 0~1.5708)
  4. joint_velocities    — URDF velocity limits (wrist 2.0x transient margin)
  5. root_height_sanity  — band [0.35, 1.0] m (F1 standing base z ≈ 0.59)
  6. continuity          — finite values, continuous root quaternion
  + jerk report (advisory, not gating)

Usage:
    python tools_check_motion_f1.py --csv motions/csv/dance1_subject3_f1.csv --fps 30
    python tools_check_motion_f1.py --csv ... --scan_windows     # find clean window
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
F1_XML = REPO / "GMR/assets/f1/f1_mocap.xml"
URDF = REPO / "F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"

F1_JOINT_NAMES = [
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

# F1 foot sole corners (ankle_roll frame) — vendor collision STL AABB,
# z_bottom = -0.0337 (see tools_build_f1_mjcf.py output / F1_AUDIT.md)
FOOT_CORNERS = {
    "left": [(-0.0967, -0.0395, -0.0337), (0.1071, -0.0395, -0.0337),
             (-0.0967, 0.0365, -0.0337), (0.1071, 0.0365, -0.0337)],
    "right": [(-0.0967, -0.0365, -0.0337), (0.1071, -0.0365, -0.0337),
              (-0.0967, 0.0395, -0.0337), (0.1071, 0.0395, -0.0337)],
}
FOOT_SPHERE_R = 0.002  # matches the sim contact spheres

# F1 leg body names lack the _link suffix
def B(name: str) -> str:
    if any(t in name for t in ("hip", "knee", "ankle")):
        return name.removesuffix("_link")
    return name

CAPSULE_R = {
    "base_link": 0.09,
    "left_hip_pitch_link": 0.055, "right_hip_pitch_link": 0.055,
    "left_hip_roll_link": 0.05, "right_hip_roll_link": 0.05,
    "left_hip_yaw_link": 0.05, "right_hip_yaw_link": 0.05,
    "left_knee_pitch_link": 0.05, "right_knee_pitch_link": 0.05,
    "left_ankle_roll_link": 0.04, "right_ankle_roll_link": 0.04,
    "left_shoulder_pitch_link": 0.05, "right_shoulder_pitch_link": 0.05,
    "left_shoulder_roll_link": 0.05, "right_shoulder_roll_link": 0.05,
    "left_shoulder_yaw_link": 0.045, "right_shoulder_yaw_link": 0.045,
    "left_elbow_pitch_link": 0.045, "right_elbow_pitch_link": 0.045,
    "left_elbow_yaw_link": 0.04, "right_elbow_yaw_link": 0.04,
    "left_wrist_roll_link": 0.02, "right_wrist_roll_link": 0.02,
}
# F1 lumbar_pitch local chest axis: +y is up in the rotated link frame (same
# construction as X1; upper-body frames identical)
TORSO_BODY = "lumbar_pitch_link"
TORSO_OFFSETS = [(0.0, 0.05, 0.0), (0.0, 0.15, 0.0), (0.0, 0.25, 0.0)]
TORSO_R = 0.075
CHECK_PAIRS = [
    ("left_wrist_roll_link", "torso_0"), ("right_wrist_roll_link", "torso_0"),
    ("left_wrist_roll_link", "torso_1"), ("right_wrist_roll_link", "torso_1"),
    ("left_wrist_roll_link", "torso_2"), ("right_wrist_roll_link", "torso_2"),
    ("left_wrist_roll_link", "base_link"), ("right_wrist_roll_link", "base_link"),
    ("left_wrist_roll_link", "right_wrist_roll_link"),
    ("left_elbow_pitch_link", "torso_2"), ("right_elbow_pitch_link", "torso_2"),
    ("left_wrist_roll_link", "right_knee_pitch_link"),
    ("right_wrist_roll_link", "left_knee_pitch_link"),
    ("left_knee_pitch_link", "right_knee_pitch_link"),
    ("left_ankle_roll_link", "right_ankle_roll_link"),
    ("left_elbow_pitch_link", "right_elbow_pitch_link"),
    ("left_shoulder_yaw_link", "right_shoulder_yaw_link"),
]


def urdf_limits():
    tree = ET.parse(URDF)
    out = {}
    for j in tree.getroot().iter("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        lim = j.find("limit")
        out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")), float(lim.get("velocity")))
    return out


def load_motion(args):
    if args.csv:
        raw = np.loadtxt(args.csv, delimiter=",")
        return raw[:, :3], raw[:, 3:7], raw[:, 7:], args.fps
    data = np.load(args.npz)
    fps = float(data["fps"])
    body_names = [str(b) for b in data["body_names"]]
    jp = data["joint_pos"]
    base_idx = body_names.index("base_link")
    root_pos = data["body_pos_w"][:, base_idx]
    root_quat = data["body_quat_w"][:, base_idx][:, [1, 2, 3, 0]]  # wxyz->xyzw
    return root_pos, root_quat, jp, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--penetration_tol", type=float, default=0.015)
    ap.add_argument("--self_clearance", type=float, default=0.005)
    ap.add_argument("--vel_margin", type=float, default=1.5)
    ap.add_argument("--height_range", type=float, nargs=2, default=(0.35, 1.0))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--json_out", type=Path, default=None)
    ap.add_argument("--scan_windows", action="store_true")
    ap.add_argument("--min_window", type=int, default=600)
    args = ap.parse_args()
    if not (args.csv or args.npz):
        ap.error("need --csv or --npz")

    root_pos, root_quat, joints, fps = load_motion(args)
    n_frames = len(joints)
    frames = range(0, n_frames, args.stride)
    print(f"motion: {n_frames} frames @ {fps} FPS (checking every {args.stride})")

    limits = urdf_limits()

    # FK model with capsule spheres injected
    tree = ET.parse(F1_XML)
    bodies = {b.get("name"): b for b in tree.getroot().iter("body")}
    for i, off in enumerate(TORSO_OFFSETS):
        g = ET.SubElement(bodies[TORSO_BODY], "geom")
        g.set("type", "sphere"); g.set("size", str(TORSO_R))
        g.set("pos", " ".join(str(v) for v in off))
        g.set("contype", "0"); g.set("conaffinity", "0")
        g.set("rgba", "0.2 0.6 0.6 0.3"); g.set("name", f"torso_{i}")
    for body_name, r in CAPSULE_R.items():
        el = bodies[B(body_name)]
        g = ET.SubElement(el, "geom")
        g.set("type", "sphere"); g.set("size", str(r))
        g.set("contype", "0"); g.set("conaffinity", "0")
        g.set("rgba", "0.2 0.6 0.2 0.3"); g.set("name", f"cap_{body_name}")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=str(F1_XML.parent))
    tree.write(tmp.name); tmp.close()
    model = mujoco.MjModel.from_xml_path(tmp.name)
    Path(tmp.name).unlink()
    data = mujoco.MjData(model)
    geom_names = {n: f"cap_{n}" for n in CAPSULE_R}
    geom_names.update({f"torso_{i}": f"torso_{i}" for i in range(len(TORSO_OFFSETS))})

    jid_of = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in F1_JOINT_NAMES}
    qadr = np.array([model.jnt_qposadr[jid_of[n]] for n in F1_JOINT_NAMES])
    jlo = np.array([limits[n][0] for n in F1_JOINT_NAMES])
    jup = np.array([limits[n][1] for n in F1_JOINT_NAMES])
    jvel = np.array([limits[n][2] for n in F1_JOINT_NAMES])

    body_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, B(n))
               for n in set(CAPSULE_R) | {"left_ankle_roll_link", "right_ankle_roll_link"}}

    min_ground, min_ground_frame = np.inf, -1
    min_self, min_self_pair, min_self_frame = np.inf, None, -1
    limit_viol = []
    root_min, root_max = np.inf, -np.inf

    quat_ok = True
    prev = root_quat[0] / np.linalg.norm(root_quat[0])
    for i in range(1, len(root_quat)):
        q = root_quat[i] / np.linalg.norm(root_quat[i])
        if np.dot(prev, q) < 0.1:
            quat_ok = False
        prev = q

    per_frame_bad = np.zeros(n_frames, dtype=bool)
    bad_reason = {"ground": np.zeros(n_frames, dtype=bool),
                  "self": np.zeros(n_frames, dtype=bool),
                  "height": np.zeros(n_frames, dtype=bool)}
    for f in frames:
        q = np.zeros(model.nq)
        mujoco.mj_resetData(model, data)
        q[:] = data.qpos
        q[0:3] = root_pos[f]
        qq = root_quat[f] / np.linalg.norm(root_quat[f])
        q[3:7] = [qq[3], qq[0], qq[1], qq[2]]
        q[qadr] = joints[f]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        for side in ("left", "right"):
            bid = body_id[f"{side}_ankle_roll_link"]
            R = data.xmat[bid].reshape(3, 3)
            for s in FOOT_CORNERS[side]:
                p = data.xpos[bid] + R @ np.array(s)
                z = p[2] - FOOT_SPHERE_R
                if z < min_ground:
                    min_ground, min_ground_frame = z, f
                if z < -args.penetration_tol:
                    per_frame_bad[f] = True
                    bad_reason["ground"][f] = True
        for a, b in CHECK_PAIRS:
            g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_names[a])
            g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_names[b])
            dist = mujoco.mj_geomDistance(model, data, g1, g2, 5.0, None)
            if dist < min_self:
                min_self, min_self_pair, min_self_frame = dist, (a, b), f
            if dist < args.self_clearance:
                per_frame_bad[f] = True
                bad_reason["self"][f] = True
        viol = np.where((joints[f] < jlo - 1e-4) | (joints[f] > jup + 1e-4))[0]
        for v in viol:
            limit_viol.append((f, F1_JOINT_NAMES[v], float(joints[f][v]), jlo[v], jup[v]))
        root_min = min(root_min, root_pos[f][2])
        root_max = max(root_max, root_pos[f][2])
        if not (args.height_range[0] <= root_pos[f][2] <= args.height_range[1]):
            per_frame_bad[f] = True
            bad_reason["height"][f] = True

    dt = 1.0 / fps
    jvelmax = np.max(np.abs(np.diff(joints, axis=0)) / dt, axis=0)
    def _vmargin(n: str) -> float:
        return 2.0 if "wrist" in n else args.vel_margin
    per_joint_lim = np.array([_vmargin(n) * limits[n][2] for n in F1_JOINT_NAMES])
    vel_bad = (np.abs(np.diff(joints, axis=0)) / dt > per_joint_lim).any(axis=1)
    per_frame_bad[:-1] |= vel_bad
    vel_exceed = [(F1_JOINT_NAMES[i], float(jvelmax[i]), jvel[i]) for i in range(len(F1_JOINT_NAMES))
                  if jvelmax[i] > _vmargin(F1_JOINT_NAMES[i]) * jvel[i]]
    finite_ok = bool(np.isfinite(joints).all() and np.isfinite(root_pos).all())

    # jerk (advisory): second-difference angular acceleration, rad/s^2
    acc = np.abs(np.diff(joints, n=2, axis=0)) / dt**2
    jerk_max = float(np.max(acc)) if len(acc) else 0.0

    if args.scan_windows:
        best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
        for i in range(n_frames):
            if not per_frame_bad[i]:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        print(f"bad frames: {int(per_frame_bad.sum())}/{n_frames} "
              f"(ground={int(bad_reason['ground'].sum())}, self={int(bad_reason['self'].sum())}, "
              f"height={int(bad_reason['height'].sum())}, vel={int(vel_bad.sum())})")
        print(f"longest clean window: [{best_start}, {best_start+best_len}) len={best_len} "
              f"({best_len/fps:.1f}s @ {fps}fps)")
        ok = best_len >= args.min_window
        print("WINDOW_OK" if ok else "WINDOW_TOO_SHORT")
        if args.json_out:
            args.json_out.write_text(json.dumps({
                "best_window": [int(best_start), int(best_start + best_len)],
                "length_frames": int(best_len), "bad_frames": int(per_frame_bad.sum()),
            }, indent=2))
        sys.exit(0 if ok else 2)

    report = {
        "frames": n_frames,
        "fps": fps,
        "ground": {"min_foot_z_m": float(min_ground), "frame": int(min_ground_frame),
                   "pass": bool(min_ground > -args.penetration_tol), "threshold_m": -args.penetration_tol},
        "self_penetration": {"min_pair_distance_m": float(min_self), "pair": min_self_pair,
                             "frame": int(min_self_frame), "pass": bool(min_self > args.self_clearance),
                             "threshold_m": args.self_clearance},
        "joint_limits": {"violations": len(limit_viol), "examples": limit_viol[:5],
                         "pass": len(limit_viol) == 0},
        "joint_velocities": {"exceeding": vel_exceed, "margin": args.vel_margin,
                             "pass": len(vel_exceed) == 0},
        "root_height": {"min_m": float(root_min), "max_m": float(root_max),
                        "band": list(args.height_range),
                        "pass": bool(args.height_range[0] <= root_min and root_max <= args.height_range[1])},
        "continuity": {"pass": bool(quat_ok and finite_ok), "quat_continuous": quat_ok, "finite": finite_ok},
        "jerk_advisory": {"max_ang_acc_rad_s2": jerk_max, "gating": False},
        "overall_pass": None,
    }
    report["overall_pass"] = bool(report["ground"]["pass"] and report["self_penetration"]["pass"]
                                  and report["joint_limits"]["pass"] and report["joint_velocities"]["pass"]
                                  and report["root_height"]["pass"] and report["continuity"]["pass"])
    print(json.dumps(report, indent=2, default=str))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["overall_pass"] else 2)


if __name__ == "__main__":
    main()
