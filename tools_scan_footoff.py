"""Scan FOOT_OFF values: measure foot IK error and worst corner depth."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "GMR"))

import mujoco
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file

BVH = REPO / "downloads/lafan1/dance1_subject3.bvh"
frames, _ = load_bvh_file(str(BVH), format="lafan1")
SAMPLE = list(range(0, 600, 40))

for FOOT_OFF in (0.010, 0.015, 0.020, 0.025, 0.030, 0.035):
    r = GeneralMotionRetargeting(actual_human_height=1.75, src_human="bvh_lafan1",
                                 tgt_robot="xyber_x1", solver="proxqp", verbose=False)
    m = r.model
    for tbl in (r.pos_offsets1, r.pos_offsets2):
        for human in ("LeftFootMod", "RightFootMod"):
            sign = 1.0 if human.startswith("Left") else -1.0
            tbl[human] = np.array([0.0, sign * FOOT_OFF, 0.0])
    init = r.configuration.data.qpos.copy()
    for jid in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if n in (None, "floating_base"):
            continue
        adr = m.jnt_qposadr[jid]
        lo, up = m.jnt_range[jid]
        init[adr] = lo + 0.15 * (up - lo)
    r.max_iter = 30
    r.configuration.update(init)

    foot_err, worst_z = [], 0.0
    for f in SAMPLE:
        r.retarget(frames[f])
        r.update_targets(frames[f])
        d = r.configuration.data
        for side, human in (("left", "LeftFootMod"), ("right", "RightFootMod")):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            foot_err.append(np.linalg.norm(d.xpos[bid] - r.scaled_human_data[human][0]))
            Rm = d.xmat[bid].reshape(3, 3)
            dy = -0.0408 if side == "left" else 0.0408
            for sx in (0.03, -0.03):
                for sz in (0.07, -0.07):
                    z = (d.xpos[bid] + Rm @ np.array([sx, dy, sz]))[2]
                    worst_z = min(worst_z, z)
    print(f"FOOT_OFF={FOOT_OFF:.3f}: foot_err mean={np.mean(foot_err):.4f}  worst_corner_z={worst_z:.4f}")
