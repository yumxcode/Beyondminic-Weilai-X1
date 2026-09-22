"""Ablation v2: foot task costs via FrameTask.set_*_cost."""
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
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def run(tag, foot_rot_w=None, foot_pos_w=None, damping=None, max_iter=30):
    r = GeneralMotionRetargeting(actual_human_height=1.75, src_human="bvh_lafan1",
                                 tgt_robot="xyber_x1", solver="proxqp", verbose=False)
    m = r.model
    patched = 0
    for mapping in (r.human_body_to_task1, r.human_body_to_task2):
        for human in ("LeftFootMod", "RightFootMod"):
            t = mapping.get(human)
            if t is None:
                continue
            if foot_rot_w is not None:
                t.set_orientation_cost(foot_rot_w)
            if foot_pos_w is not None:
                t.set_position_cost(foot_pos_w)
            patched += 1
    if damping is not None:
        r.damping = damping
    init = r.configuration.data.qpos.copy()
    for jid in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if n in (None, "floating_base"):
            continue
        adr = m.jnt_qposadr[jid]
        lo, up = m.jnt_range[jid]
        init[adr] = lo + 0.15 * (up - lo)
    r.max_iter = max_iter
    r.configuration.update(init)

    foot_err = []
    for f in SAMPLE:
        r.retarget(frames[f])
        r.update_targets(frames[f])
        d = r.configuration.data
        for side, human in (("left", "LeftFootMod"), ("right", "RightFootMod")):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            foot_err.append(np.linalg.norm(d.xpos[bid] - r.scaled_human_data[human][0]))
    print(f"{tag} (patched {patched}): foot_err mean={np.mean(foot_err):.4f} max={np.max(foot_err):.4f}")


run("baseline", foot_rot_w=None)
run("foot rot_w=0", foot_rot_w=1e-8)
run("foot pos_w=500", foot_pos_w=500.0)
run("damping=0.05", damping=0.05)
run("max_iter=80", max_iter=80)
