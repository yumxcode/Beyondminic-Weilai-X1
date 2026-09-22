#!/usr/bin/env python3
"""Smoke test: GMR retargeting of a synthetic standing human to xyber_x1."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "GMR"))

from general_motion_retargeting import GeneralMotionRetargeting  # noqa: E402

H = {
    "Hips":          ((0.00, 0.00, 0.95), [1, 0, 0, 0]),
    "Spine2":        ((0.02, 0.00, 1.25), [1, 0, 0, 0]),
    "LeftUpLeg":     ((0.00, 0.10, 0.90), [1, 0, 0, 0]),
    "RightUpLeg":    ((0.00, -0.10, 0.90), [1, 0, 0, 0]),
    "LeftLeg":       ((0.00, 0.10, 0.50), [1, 0, 0, 0]),
    "RightLeg":      ((0.00, -0.10, 0.50), [1, 0, 0, 0]),
    "LeftFootMod":   ((0.02, 0.10, 0.08), [1, 0, 0, 0]),
    "RightFootMod":  ((0.02, -0.10, 0.08), [1, 0, 0, 0]),
    "LeftArm":       ((0.02, 0.20, 1.40), [1, 0, 0, 0]),
    "RightArm":      ((0.02, -0.20, 1.40), [1, 0, 0, 0]),
    "LeftForeArm":   ((0.02, 0.45, 1.35), [1, 0, 0, 0]),
    "RightForeArm":  ((0.02, -0.45, 1.35), [1, 0, 0, 0]),
    "LeftHand":      ((0.02, 0.65, 1.30), [1, 0, 0, 0]),
    "RightHand":     ((0.02, -0.65, 1.30), [1, 0, 0, 0]),
}


def main() -> None:
    retargeter = GeneralMotionRetargeting(
        actual_human_height=1.75,
        src_human="bvh_lafan1",
        tgt_robot="xyber_x1",
        solver="proxqp",
        verbose=False,
    )
    human_data = {k: (np.array(v[0], dtype=float), np.array(v[1], dtype=float)) for k, v in H.items()}

    # start slightly inside joint limits (knee/elbow ranges start at 0)
    init = retargeter.configuration.data.qpos.copy()
    for jid in range(retargeter.model.njnt):
        name = mujoco.mj_id2name(retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in (None, "floating_base"):
            continue
        adr = retargeter.model.jnt_qposadr[jid]
        lo, up = retargeter.model.jnt_range[jid]
        if lo < up:
            init[adr] = lo + 0.15 * (up - lo)
    retargeter.configuration.update(init)

    qpos = retargeter.retarget(human_data)
    assert qpos.shape == (36,), qpos.shape

    m = retargeter.model
    d = retargeter.configuration.data
    print("\nretargeted joints (deg) vs limits:")
    viol = 0
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in (None, "floating_base"):
            continue
        qadr = m.jnt_qposadr[jid]
        val = qpos[qadr]
        lo, up = m.jnt_range[jid]
        ok = lo - 1e-3 <= val <= up + 1e-3
        viol += 0 if ok else 1
        print(f"  {name:30s} {np.degrees(val):8.2f}  [{np.degrees(lo):7.2f}, {np.degrees(up):7.2f}]  {'ok' if ok else 'VIOLATION'}")

    for b in ["base_link", "lumbar_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"{b:24s} pos={np.round(d.xpos[bid], 3)}")
    print(f"\nlimit violations: {viol}")
    print("SMOKE_OK" if viol == 0 else "SMOKE_FAIL")


if __name__ == "__main__":
    main()
