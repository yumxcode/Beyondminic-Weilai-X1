#!/usr/bin/env python3
"""Empirically calibrate X1 IK config offsets against LAFAN dance1_subject3.

Greedy search per body group:
  - FOOT_OFF in a small grid (foot position error)
  - shoulder/elbow/wrist rot_offsets: right-multiplied by candidates from the
    24-element octahedral rotation group (arm position errors)

Uses the actual GMR IK in the loop; a fixed initialization keeps comparisons
fair. Writes the best config back to bvh_lafan1_to_xyber_x1.json.
"""
import itertools
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "GMR"))

from general_motion_retargeting import GeneralMotionRetargeting  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

BVH = REPO / "downloads/lafan1/dance1_subject3.bvh"
CFG = REPO / "GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_xyber_x1.json"
SAMPLE = list(range(0, 300, 50))  # 6 frames

# octahedral group: 24 rotations aligning axis-permutation frames
AXES = np.eye(3)
CANDS = []
for perm in itertools.permutations(range(3)):
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                M = np.zeros((3, 3))
                M[0, perm[0]] = sx
                M[1, perm[1]] = sy
                M[2, perm[2]] = sz
                if round(np.linalg.det(M)) == 1:
                    CANDS.append(R.from_matrix(M))
# dedupe
uniq = []
for c in CANDS:
    if not any(np.allclose(c.as_matrix(), u.as_matrix()) for u in uniq):
        uniq.append(c)
CANDS = uniq
print(f"{len(CANDS)} candidate corrections")


def fresh_retargeter():
    r = GeneralMotionRetargeting(actual_human_height=1.75, src_human="bvh_lafan1",
                                 tgt_robot="xyber_x1", solver="proxqp", verbose=False)
    m = r.model
    init = r.configuration.data.qpos.copy()
    for jid in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if n in (None, "floating_base"):
            continue
        adr = m.jnt_qposadr[jid]
        lo, up = m.jnt_range[jid]
        init[adr] = lo + 0.15 * (up - lo)
    r.max_iter = 30
    return r


def body_pos(r, name):
    bid = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return r.configuration.data.xpos[bid].copy()


def evaluate(overrides=None, foot_off=None, frames=None, sample=SAMPLE):
    """overrides: {(side, group): Rotation}; foot_off: float or None"""
    r = fresh_retargeter()
    if foot_off is not None:
        # patch pos_offsets for feet (table1 and table2)
        for tbl in (r.pos_offsets1, r.pos_offsets2):
            tbl["LeftFootMod" if "LeftFootMod" in tbl else "left_foot_index"]
    # patch rot offsets through the loaded config objects
    if overrides:
        for (human_name, side), corr in overrides.items():
            for tbl in (r.rot_offsets1, r.rot_offsets2):
                if human_name in tbl:
                    tbl[human_name] = tbl[human_name] * corr
    frames_local, _ = load_bvh_file(str(BVH), format="lafan1") if frames is None else (frames, None)
    m = r.model
    init = r.configuration.data.qpos.copy()
    for jid in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if n in (None, "floating_base"):
            continue
        adr = m.jnt_qposadr[jid]
        lo, up = m.jnt_range[jid]
        init[adr] = lo + 0.15 * (up - lo)
    r.configuration.update(init)

    errs = {}
    for f in sample:
        r.retarget(frames_local[f])
        r.update_targets(frames_local[f])
        tgt = r.scaled_human_data
        for side, pref in (("left", "Left"), ("right", "Right")):
            pairs = {
                "shoulder": (f"{pref}Arm", f"{side}_shoulder_pitch_link"),
                "elbow": (f"{pref}ForeArm", f"{side}_elbow_pitch_link"),
                "wrist": (f"{pref}Hand", f"{side}_wrist_roll_link"),
                "foot": (f"{pref}FootMod", f"{side}_ankle_roll_link"),
            }
            for gname, (human, body) in pairs.items():
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
                errs.setdefault(gname, []).append(np.linalg.norm(r.configuration.data.xpos[bid] - tgt[human][0]))
    return {k: float(np.mean(v)) for k, v in errs.items()}


def main():
    frames, _ = load_bvh_file(str(BVH), format="lafan1")

    base = evaluate(frames=frames)
    print("baseline:", {k: round(v, 4) for k, v in base.items()})

    # ---- greedy arm calibration (same correction for L and R of a group) ----
    best_overrides = {}
    human_names = {"shoulder": ("LeftArm", "RightArm"),
                   "elbow": ("LeftForeArm", "RightForeArm"),
                   "wrist": ("LeftHand", "RightHand")}

    for group in ("shoulder", "elbow", "wrist"):
        best_err, best_corr = None, None
        downstream = {"shoulder": ["shoulder", "elbow", "wrist"],
                      "elbow": ["elbow", "wrist"],
                      "wrist": ["wrist"]}[group]
        for i, corr in enumerate(CANDS + [R.identity()]):
            ov = {**{(human_names[g][0], g): corr for g in (group,)},
                  **{(human_names[g][1], g): corr for g in (group,)}}
            ov.update({k: v for k, v in best_overrides.items() if k[1] != group})
            try:
                e = evaluate(overrides=ov, frames=frames)
            except Exception:
                continue
            score = sum(e[g] for g in downstream)
            if best_err is None or score < best_err:
                best_err, best_corr = score, corr
        if best_corr is not None and not np.allclose(best_corr.as_matrix(), np.eye(3)):
            for side_human in human_names[group]:
                best_overrides[(side_human, group)] = best_corr
            print(f"{group}: correction={np.round(best_corr.as_quat(scalar_first=True),3)} score={best_err:.4f}")

    final = evaluate(overrides=best_overrides or None, frames=frames)
    print("calibrated:", {k: round(v, 4) for k, v in final.items()})

    if best_overrides:
        cfg = json.load(open(CFG))
        for (human_name, group), corr in best_overrides.items():
            q = corr.as_quat(scalar_first=True)
            for tbl in ("ik_match_table1", "ik_match_table2"):
                for body, entry in cfg[tbl].items():
                    if entry[0] == human_name:
                        old = R.from_quat(np.array(entry[4]) / np.linalg.norm(entry[4]), scalar_first=True)
                        newq = (old * corr).as_quat(scalar_first=True)
                        entry[4] = [float(round(v, 10)) for v in newq]
        CFG.write_text(json.dumps(cfg, indent=4))
        print(f"wrote {CFG}")


if __name__ == "__main__":
    main()
