#!/usr/bin/env python3
"""Lift ground-penetrating frames of an F1 retarget CSV out of the floor.

Post-retarget fix for scaled-foot artifacts: the F1 sole (20.4 cm heel->toe
AABB) is longer than the human foot the GMR ground normalization used, so a
hard step-landing frame can dip a sole corner 1-2 cm below z=0 even though the
source clip never penetrates. IK joints are untouched — only the root z gets a
smoothly-windowed bump on the offending frames, so foot posture (gate 7) and
arm quality (gate 8) are unaffected.

  per-frame lift = max(0, -sole_min_z - target_depth)   (only where deeper
                   than target_depth, default -0.013 = just inside the gate)
  lift is Gaussian-smoothed (sigma frames) so the root trajectory stays C1,
  then added to root z.  Optional --start_frame drops leading IK-settling
  frames (retarget started mid-motion; first few frames converge from the
  init-margin pose and can dip 3+ cm).

Usage:
    python tools_lift_ground_f1.py --csv in.csv --out out.csv --start_frame 7
    python tools_lift_ground_f1.py --csv in.csv --out out.csv --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from tools_check_motion_f1 import (  # noqa: E402
    F1_JOINT_NAMES, F1_XML, B, FOOT_CORNERS, FOOT_SPHERE_R,
)


def sole_min_z_per_frame(m: np.ndarray, model, data, qadr) -> np.ndarray:
    bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, B(f"{s}_ankle_roll_link"))
           for s in ("left", "right")}
    out = np.zeros(len(m))
    for f in range(len(m)):
        mujoco.mj_resetData(model, data)
        q = data.qpos.copy()
        q[0:3] = m[f, 0:3]
        qq = m[f, 3:7] / np.linalg.norm(m[f, 3:7])
        q[3:7] = [qq[3], qq[0], qq[1], qq[2]]
        q[qadr] = m[f, 7:]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        z = np.inf
        for s in ("left", "right"):
            R = data.xmat[bid[s]].reshape(3, 3)
            p = data.xpos[bid[s]]
            for c in FOOT_CORNERS[s]:
                z = min(z, (p + R @ np.array(c))[2] - FOOT_SPHERE_R)
        out[f] = z
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start_frame", type=int, default=0,
                    help="drop this many leading frames (IK settling transient)")
    ap.add_argument("--target_depth", type=float, default=-0.013,
                    help="lift until every frame's sole min z >= this (m)")
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="Gaussian smoothing sigma for the lift profile (frames)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(F1_XML))
    data = mujoco.MjData(model)
    jid_of = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in F1_JOINT_NAMES}
    qadr = np.array([model.jnt_qposadr[jid_of[n]] for n in F1_JOINT_NAMES])

    m = np.loadtxt(args.csv, delimiter=",")
    if args.start_frame:
        m = m[args.start_frame:].copy()
        print(f"[trim] dropped first {args.start_frame} frames -> {len(m)}")

    z0 = sole_min_z_per_frame(m, model, data, qadr)
    lift_raw = np.maximum(0.0, -(z0 - args.target_depth))
    n_lift = int((lift_raw > 0).sum())
    print(f"[ground] sole min z before: {z0.min()*1000:.1f} mm, frames needing lift: {n_lift}")

    if n_lift == 0:
        print("[ground] nothing to lift")
    else:
        # gaussian smoothing keeps root z C1 across the bump edges
        x = np.arange(-4 * int(np.ceil(args.sigma)), 4 * int(np.ceil(args.sigma)) + 1)
        kernel = np.exp(-0.5 * (x / args.sigma) ** 2)
        kernel /= kernel.sum()
        lift = np.convolve(lift_raw, kernel, mode="same")
        # the smoothed profile can dip below the needed lift at the worst
        # frames; enforce the target by taking a running max-blend
        lift = np.maximum(lift, lift_raw)
        m[:, 2] += lift
        z1 = sole_min_z_per_frame(m, model, data, qadr)
        print(f"[ground] sole min z after : {z1.min()*1000:.1f} mm "
              f"(max lift applied: {lift.max()*1000:.1f} mm)")
        if z1.min() < args.target_depth - 1e-6:
            raise SystemExit(f"[ground] lift insufficient: {z1.min():.4f} < {args.target_depth}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, m, delimiter=",", fmt="%.8f")
    print(f"[saved] {args.out} ({len(m)} frames)")


if __name__ == "__main__":
    main()
