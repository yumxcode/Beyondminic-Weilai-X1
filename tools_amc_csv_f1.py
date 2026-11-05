#!/usr/bin/env python3
"""Retarget a CMU ASF/AMC trial (via LAFAN-semantic frames) to the F1 CSV.

Combines tools_amc_to_lafan_frames.py (ASF/AMC parse + FK + skeleton
alignment to LAFAN semantics) with the exact IK loop validated in
tools_bvh_to_csv_f1.py (same retargeter config: bvh_lafan1_to_f1.json,
solver proxqp, init_margin 0.15, freeze-frame fallback, clamp, smooth).

Output: 36-column CSV identical in layout to tools_bvh_to_csv_f1.py
(root_pos_xyz, root_quat_xyzw, 29 canonical-order joints).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from tools_amc_to_lafan_frames import (
    ASF_LEN_SCALE,
    DEFAULT_GMR_ROOT,
    _to_gmr,
    align_to_lafan,
    convert_frames,
    fk_skeleton,
    lafan_reference_frame,
    parse_amc,
    parse_asf,
)
from tools_bvh_to_csv_f1 import (
    F1_JOINT_NAMES,
    make_quaternions_continuous,
    smooth_motion,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asf", type=Path, required=True)
    ap.add_argument("--amc", type=Path, required=True)
    ap.add_argument("--output_file", type=Path, required=True)
    ap.add_argument("--target_fps", type=int, default=30)
    ap.add_argument("--start_frame", type=int, default=0, help="AMC frame index (120fps)")
    ap.add_argument("--end_frame", type=int, default=None)
    ap.add_argument("--solver", default="proxqp")
    ap.add_argument("--init_margin", type=float, default=0.15)
    ap.add_argument("--smooth_window", type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, str(DEFAULT_GMR_ROOT))
    import mujoco as mj
    from general_motion_retargeting import GeneralMotionRetargeting

    asf = parse_asf(args.asf)
    amc = parse_amc(args.amc)
    end = len(amc) if args.end_frame is None else args.end_frame
    stride = 120 // args.target_fps
    src_idx = list(range(args.start_frame, end, stride))
    print(f"[src] {len(src_idx)} frames @{args.target_fps}fps from {len(amc)} AMC frames")

    # skeleton alignment (auto stance)
    lafan_stance = lafan_reference_frame(DEFAULT_GMR_ROOT,
                                         REPO / "downloads/lafan1/dance1_subject3.bvh")
    pos = np.array([fk_skeleton(asf, amc[i])["root"][0][:2] for i in src_idx])
    w = max(2, len(pos) // 10)
    moves = np.convolve(np.linalg.norm(np.diff(pos, axis=0), axis=1), np.ones(w), "valid")
    stance_out = int(np.argmin(moves)) + w // 2
    stance_frame = src_idx[min(stance_out, len(src_idx) - 1)]
    print(f"[align] stance AMC frame {stance_frame}")
    cmu_stance = fk_skeleton(asf, amc[stance_frame])
    R_yaw, B = align_to_lafan(cmu_stance, lafan_stance)

    # ground normalization: the CMU world ground plane is not z=0 (measured
    # ~ -0.08 m); shift so the lowest foot/toe contact over the clip is z=0
    zs = []
    for i in src_idx:
        fk = fk_skeleton(asf, amc[i])
        for bone in ("lfoot", "rfoot", "ltoes", "rtoes"):
            p, _ = _to_gmr(*fk[bone])
            zs.append(p[2])
    ground_z = -min(zs)
    print(f"[ground] lowest foot point {min(zs):.3f} m -> global shift +{ground_z:.3f} m")

    frames = convert_frames(asf, amc, R_yaw, B, src_idx, ground_z=ground_z)
    travel = np.linalg.norm(frames[-1]["Hips"][0][:2] - frames[0]["Hips"][0][:2])
    print(f"[src] Hips travel {travel:.2f} m over {len(frames)/args.target_fps:.1f}s")

    retargeter = GeneralMotionRetargeting(
        actual_human_height=1.75,
        src_human="bvh_lafan1",
        tgt_robot="f1",
        solver=args.solver,
        verbose=False,
        use_velocity_limit=False,
    )

    # --- segment-length re-scaled IK targets ---------------------------------
    # The committed human_scale_table is calibrated for the LAFAN skeleton
    # (leg 0.831 m etc.). The CMU subject has different proportions
    # (leg ~0.93 m): using the LAFAN ratios puts the root-height target above
    # F1's leg reach and drives systematic foot penetration. Recompute the
    # three segment scales from the F1 mocap model and this ASF's T-pose.
    def _f1_seg_lengths():
        m = retargeter.model
        d = mj.MjData(m)
        mj.mj_resetData(m, d)
        mj.mj_forward(m, d)

        def p(name):
            bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
            assert bid >= 0, f"body {name} missing in F1 mocap model"
            return d.xpos[bid]
        # upper-body bodies carry the _link suffix in the F1 mocap xml
        leg = np.linalg.norm(p("left_ankle_roll") - p("left_hip_pitch"))
        arm = np.linalg.norm(p("left_wrist_roll_link") - p("left_shoulder_pitch_link"))
        spine = np.linalg.norm(p("lumbar_pitch_link") - p("base_link"))
        return leg, arm, spine

    def _cmu_seg_lengths():
        # bone lengths from the ASF :length fields (rest-pose independent —
        # the 137 ASF rest pose has bent/crouched limbs, so joint positions
        # would give wrong segment lengths)
        def blen(name):
            return asf["bones"][name]["length"] * ASF_LEN_SCALE
        leg = blen("lfemur") + blen("ltibia")
        arm = blen("lhumerus") + blen("lradius")
        spine = blen("lowerback") + blen("upperback") + blen("thorax")
        return leg, arm, spine

    f1_leg, f1_arm, f1_spine = _f1_seg_lengths()
    cmu_leg, cmu_arm, cmu_spine = _cmu_seg_lengths()
    leg_s, arm_s, spine_s = f1_leg / cmu_leg, f1_arm / cmu_arm, f1_spine / cmu_spine
    for k in list(retargeter.human_scale_table.keys()):
        if any(t in k for t in ("Leg", "Foot", "Hips")):
            retargeter.human_scale_table[k] = leg_s
        elif any(t in k for t in ("Arm", "Hand")):
            retargeter.human_scale_table[k] = arm_s
        else:
            retargeter.human_scale_table[k] = spine_s
    print(f"[scale] CMU-measured leg {cmu_leg:.3f} arm {cmu_arm:.3f} spine {cmu_spine:.3f} "
          f"-> scales {leg_s:.3f}/{arm_s:.3f}/{spine_s:.3f}")

    model = retargeter.model
    qpos_adrs, limits = [], []
    for name in F1_JOINT_NAMES:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        limits.append(model.jnt_range[jid].copy())
    qpos_adrs = np.array(qpos_adrs)
    lo = np.array([l[0] for l in limits])
    up = np.array([l[1] for l in limits])

    init = retargeter.configuration.data.qpos.copy()
    for adr, l, u in zip(qpos_adrs, lo, up):
        init[adr] = l + args.init_margin * (u - l)
    retargeter.configuration.update(init)
    retargeter.max_iter = 30

    qpos_frames, failed = [], []
    for index, frame in enumerate(frames):
        try:
            qpos = np.asarray(retargeter.retarget(frame), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — freeze-frame fallback
            print(f"[warn] frame {index}: retarget failed ({type(exc).__name__}), freezing pose")
            failed.append(index)
            qpos = qpos_frames[-1].copy() if qpos_frames else None
            if qpos is None:
                continue
        if qpos.shape != (36,):
            raise RuntimeError(f"expected qpos (36,), got {qpos.shape} at frame {index}")
        qpos[qpos_adrs] = np.clip(qpos[qpos_adrs], lo + 0.01, up - 0.01)
        qpos_frames.append(qpos)
        if index % 100 == 0:
            print(f"frame {index}/{len(frames)}")

    motion = np.stack(qpos_frames).astype(np.float32)
    if not np.all(np.isfinite(motion)):
        raise RuntimeError("non-finite value in retargeted motion")

    if args.smooth_window > 1:
        motion = smooth_motion(motion.astype(np.float64), args.smooth_window).astype(np.float32)
        motion[:, qpos_adrs] = np.clip(motion[:, qpos_adrs], lo + 0.01 - 1e-6, up - 0.01 + 1e-6)
        print(f"[smooth] window={args.smooth_window}")

    motion[:, 3:7] = motion[:, [4, 5, 6, 3]]  # wxyz -> xyzw
    motion[:, 3:7] = make_quaternions_continuous(motion[:, 3:7])

    reordered = np.zeros_like(motion)
    reordered[:, 0:7] = motion[:, 0:7]
    reordered[:, 7:] = motion[:, qpos_adrs]
    motion = reordered

    out = args.output_file.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, motion, delimiter=",", fmt="%.8f")
    print(f"Saved {len(motion)} frames ({(len(motion)-1)/args.target_fps:.2f}s) to {out}; "
          f"frozen: {len(failed)}")


if __name__ == "__main__":
    main()
