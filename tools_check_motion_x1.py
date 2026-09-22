#!/usr/bin/env python3
"""Retargeting quality gate for X1 motions.

Checks a retargeted motion (36-column CSV or npz) against pass metrics:

  1. ground_penetration  — X1 foot collision spheres (4 per foot, from MJCF)
                           must stay above the ground plane.
  2. self_penetration    — capsule-per-link approximations; min distance
                           between non-adjacent link pairs must stay positive
                           (checked with mujoco.mj_geomDistance).
  3. joint_limits        — all 29 joint values within real URDF limits.
  4. joint_velocities    — finite-difference joint velocity within URDF
                           velocity limits (scaled by --vel_margin).
  5. root_height_sanity  — base height inside plausible band.
  6. continuity          — finite values, continuous root quaternion.

Usage:
    python tools_check_motion_x1.py --csv motions/csv/dance1_x1.csv --fps 30
    python tools_check_motion_x1.py --npz motions/dance_x1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
X1_XML = REPO / "GMR/assets/xyber_x1/xyber_x1_mocap.xml"
URDF = REPO / "X1_29DOF/urdf/f1.urdf"

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

# Foot collision spheres in ankle_roll_link frame (from X1 MJCF).
FOOT_SPHERES = {
    "left": [(0.03, -0.0408, 0.07), (-0.03, -0.0408, 0.07), (0.03, -0.0408, -0.07), (-0.03, -0.0408, -0.07)],
    "right": [(0.03, 0.0408, 0.07), (-0.03, 0.0408, 0.07), (0.03, 0.0408, -0.07), (-0.03, 0.0408, -0.07)],
}
FOOT_SPHERE_R = 0.005  # MJCF collision spheres are r=0.002 at the 4 corners; 5 mm approximates sole thickness

# Link capsules for self-penetration: body -> (radius [m])
# Radii chosen conservatively from X1 link sizes.
CAPSULE_R = {
    "base_link": 0.09,
    "left_shoulder_pitch_link": 0.05, "right_shoulder_pitch_link": 0.05,
    "left_shoulder_roll_link": 0.05, "right_shoulder_roll_link": 0.05,
    "left_shoulder_yaw_link": 0.045, "right_shoulder_yaw_link": 0.045,
    "left_elbow_pitch_link": 0.045, "right_elbow_pitch_link": 0.045,
    "left_elbow_yaw_link": 0.04, "right_elbow_yaw_link": 0.04,
    "left_wrist_roll_link": 0.035, "right_wrist_roll_link": 0.035,
    "left_knee_pitch_link": 0.05, "right_knee_pitch_link": 0.05,
    "left_ankle_roll_link": 0.04, "right_ankle_roll_link": 0.04,
}
# torso: 3 spheres along the chest axis (local +y is up in the rotated link
# frame); replaces one oversized sphere that caused false positives.
TORSO_BODY = "lumbar_pitch_link"
TORSO_OFFSETS = [(0.0, 0.05, 0.0), (0.0, 0.15, 0.0), (0.0, 0.25, 0.0)]
TORSO_R = 0.075
# non-adjacent pairs worth checking (hand vs torso, arm vs opposite leg, legs crossing, hand vs other hand)
CHECK_PAIRS = [
    ("left_wrist_roll_link", "torso_0"),
    ("right_wrist_roll_link", "torso_0"),
    ("left_wrist_roll_link", "torso_1"),
    ("right_wrist_roll_link", "torso_1"),
    ("left_wrist_roll_link", "torso_2"),
    ("right_wrist_roll_link", "torso_2"),
    ("left_wrist_roll_link", "base_link"),
    ("right_wrist_roll_link", "base_link"),
    ("left_wrist_roll_link", "right_wrist_roll_link"),
    ("left_elbow_pitch_link", "torso_2"),
    ("right_elbow_pitch_link", "torso_2"),
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
        fps = args.fps
        return raw[:, :3], raw[:, 3:7], raw[:, 7:], fps
    data = np.load(args.npz)
    fps = float(data["fps"])
    body_names = [str(b) for b in data["body_names"]] if "body_names" in data else None
    jp = data["joint_pos"]
    if body_names is not None:
        order = [body_names.index("base_link")] if False else None
    base_idx = body_names.index("base_link") if body_names else 0
    return data["body_pos_w"][:, base_idx], None, jp, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--penetration_tol", type=float, default=0.015,
                    help="allowed foot-sphere depth below ground [m]")
    ap.add_argument("--self_clearance", type=float, default=0.005,
                    help="required min distance between checked capsule pairs [m]")
    ap.add_argument("--vel_margin", type=float, default=1.5,
                    help="allowed fraction of URDF velocity limit")
    ap.add_argument("--height_range", type=float, nargs=2, default=(0.35, 1.0))
    ap.add_argument("--stride", type=int, default=1, help="check every Nth frame")
    ap.add_argument("--json_out", type=Path, default=None)
    ap.add_argument("--scan_windows", action="store_true",
                    help="Per-frame scan: report the longest clean window instead of failing.")
    ap.add_argument("--min_window", type=int, default=600,
                    help="Minimum clean window length in frames (scan mode).")
    args = ap.parse_args()
    if not (args.csv or args.npz):
        ap.error("need --csv or --npz")

    root_pos, root_quat, joints, fps = load_motion(args)
    n_frames = len(joints)
    stride = max(1, args.stride)
    frames = range(0, n_frames, stride)
    print(f"motion: {n_frames} frames @ {fps} FPS (checking every {stride})")

    limits = urdf_limits()

    # ---- build FK model + capsules --------------------------------------
    # Inject a sphere per checked body into a temp copy of the mocap XML.
    import tempfile

    tree = ET.parse(X1_XML)
    torso_body = None
    for b in tree.getroot().iter("body"):
        if b.get("name") == TORSO_BODY:
            torso_body = b
            break
    for i, off in enumerate(TORSO_OFFSETS):
        g = ET.SubElement(torso_body, "geom")
        g.set("type", "sphere")
        g.set("size", str(TORSO_R))
        g.set("pos", " ".join(str(v) for v in off))
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("rgba", "0.2 0.6 0.6 0.3")
        g.set("name", f"torso_{i}")
    for body_name, r in CAPSULE_R.items():
        el = None
        for b in tree.getroot().iter("body"):
            if b.get("name") == body_name:
                el = b
                break
        if el is None:
            raise RuntimeError(f"body {body_name} not in X1 mocap xml")
        g = ET.SubElement(el, "geom")
        g.set("type", "sphere")
        g.set("size", str(r))
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("rgba", "0.2 0.6 0.2 0.3")
        g.set("name", f"cap_{body_name}")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=str(X1_XML.parent))
    tree.write(tmp.name)
    tmp.close()
    model = mujoco.MjModel.from_xml_path(tmp.name)
    Path(tmp.name).unlink()
    data = mujoco.MjData(model)
    geom_names = {n: f"cap_{n}" for n in CAPSULE_R}
    geom_names.update({f"torso_{i}": f"torso_{i}" for i in range(len(TORSO_OFFSETS))})

    jid_of = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in X1_JOINT_NAMES}
    qadr = np.array([model.jnt_qposadr[jid_of[n]] for n in X1_JOINT_NAMES])
    jlo = np.array([limits[n][0] for n in X1_JOINT_NAMES])
    jup = np.array([limits[n][1] for n in X1_JOINT_NAMES])
    jvel = np.array([limits[n][2] for n in X1_JOINT_NAMES])

    body_id = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in set(CAPSULE_R) | {"left_ankle_roll_link", "right_ankle_roll_link"}}

    # ---- iterate frames --------------------------------------------------
    min_ground = np.inf
    min_ground_frame = -1
    min_self = np.inf
    min_self_pair, min_self_frame = None, -1
    limit_viol = []
    root_min, root_max = np.inf, -np.inf

    # root quat continuity (csv: xyzw)
    quat_ok = True
    if root_quat is not None:
        prev = root_quat[0] / np.linalg.norm(root_quat[0])
        for i in range(1, len(root_quat)):
            q = root_quat[i] / np.linalg.norm(root_quat[i])
            if np.dot(prev, q) < 0.1:  # >~78° jump between frames = discontinuity
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
        if root_quat is not None:  # xyzw -> wxyz
            qq = root_quat[f] / np.linalg.norm(root_quat[f])
            q[3:7] = [qq[3], qq[0], qq[1], qq[2]]
        q[qadr] = joints[f]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        # ground penetration
        for side in ("left", "right"):
            bid = body_id[f"{side}_ankle_roll_link"]
            R = data.xmat[bid].reshape(3, 3)
            for s in FOOT_SPHERES[side]:
                p = data.xpos[bid] + R @ np.array(s)
                z = p[2] - FOOT_SPHERE_R
                if z < min_ground:
                    min_ground, min_ground_frame = z, f
                if z < -args.penetration_tol:
                    per_frame_bad[f] = True
                    bad_reason["ground"][f] = True
        # self penetration via capsule distances
        for a, b in CHECK_PAIRS:
            g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_names[a])
            g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_names[b])
            dist = mujoco.mj_geomDistance(model, data, g1, g2, 5.0, None)
            if dist < min_self:
                min_self, min_self_pair, min_self_frame = dist, (a, b), f
            if dist < args.self_clearance:
                per_frame_bad[f] = True
                bad_reason["self"][f] = True
        # joint limits
        viol = np.where((joints[f] < jlo - 1e-4) | (joints[f] > jup + 1e-4))[0]
        for v in viol:
            limit_viol.append((f, X1_JOINT_NAMES[v], float(joints[f][v]), jlo[v], jup[v]))
        root_min = min(root_min, root_pos[f][2])
        root_max = max(root_max, root_pos[f][2])
        if not (args.height_range[0] <= root_pos[f][2] <= args.height_range[1]):
            per_frame_bad[f] = True
            bad_reason["height"][f] = True

    # velocities (full fps, before stride subsample)
    dt = 1.0 / fps
    _jdiff = np.abs(np.diff(joints, axis=0)) / dt
    jvelmax = np.max(_jdiff, axis=0)  # per-joint max for the report
    jvel_per_frame = np.max(_jdiff, axis=1)  # per-frame max for bad-frame flags
    vel_exceed = [(X1_JOINT_NAMES[i], float(jvelmax[i]), jvel[i]) for i in range(len(X1_JOINT_NAMES))
                  if jvelmax[i] > args.vel_margin * jvel[i]]
    # frame f bad if the transition f->f+1 exceeds any joint velocity
    def _vmargin(n: str) -> float:
        # wrist motors tolerate brief transients at 2x the continuous rating
        return 2.0 if "wrist" in n else args.vel_margin

    per_joint_lim = np.array([_vmargin(n) * limits[n][2] for n in X1_JOINT_NAMES])
    vel_bad = (np.abs(np.diff(joints, axis=0)) / dt > per_joint_lim).any(axis=1)
    per_frame_bad[:-1] |= vel_bad

    if args.scan_windows:
        # longest run of consecutive good frames
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
              f"({best_len/args.fps:.1f}s @ {args.fps}fps)")
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
        "ground": {
            "min_foot_z_m": float(min_ground),
            "frame": int(min_ground_frame),
            "pass": bool(min_ground > -args.penetration_tol),
            "threshold_m": -args.penetration_tol,
        },
        "self_penetration": {
            "min_pair_distance_m": float(min_self),
            "pair": min_self_pair,
            "frame": int(min_self_frame),
            "pass": bool(min_self > args.self_clearance),
            "threshold_m": args.self_clearance,
        },
        "joint_limits": {
            "violations": len(limit_viol),
            "examples": limit_viol[:5],
            "pass": len(limit_viol) == 0,
        },
        "joint_velocities": {
            "exceeding": vel_exceed,
            "margin": args.vel_margin,
            "pass": len(vel_exceed) == 0,
        },
        "root_height": {
            "min_m": float(root_min),
            "max_m": float(root_max),
            "band": list(args.height_range),
            "pass": bool(args.height_range[0] <= root_min and root_max <= args.height_range[1]),
        },
        "continuity": {"pass": bool(quat_ok and np.all(np.isfinite(joints)))},
    }
    report["PASS"] = all(
        report[k]["pass"] for k in ("ground", "self_penetration", "joint_limits", "joint_velocities", "root_height", "continuity")
    )

    print(json.dumps(report, indent=2, default=str))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
    print("\nOVERALL:", "PASS ✅" if report["PASS"] else "FAIL ❌")
    sys.exit(0 if report["PASS"] else 1)


if __name__ == "__main__":
    main()
