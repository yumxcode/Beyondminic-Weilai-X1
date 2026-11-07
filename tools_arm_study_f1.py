#!/usr/bin/env python3
"""Arm-retarget variant study for F1 (mirrors X1 x1_arm_study.py approach).

Baseline (bvh_lafan1_to_f1.json) pins wrists at their clamp limit 100% of
frames and saturates R elbow/shoulder (see tools_diag_foot_f1.py baseline).
Root cause: the scaled human hand target sits at/past the F1 arm reachable
boundary (p95 0.415 m vs reach 0.421 m), so whole-body IK pushes every arm
joint against its limit. Softening IK weights alone does NOT fix it (v1/v2);
the target itself must be made reachable.

Variants:
  v0_baseline     : original config (reproduce saturation)
  v1_soft_wrist   : wrist rot_weight 10 -> 1 (both tables)
  v2_soft_arm     : v1 + elbow rot 10 -> 3, shoulder rot 10 -> 5
  v3_arm_relock   : whole-body solve, then per-arm secondary IK: freeze
                    non-arm dofs, re-anchor the wrist target to the robot's
                    FK shoulder and clamp it into 0.92*reach, solve with a
                    posture regularizer toward a LOW-PASS of the solved arm
                    pose (keeps motion alive instead of pinning joints at
                    range center). Parameterized: v3_c{posture}_r{reach},
                    v3lp_alpha{a}.

Scores: saturation %, ROM utilization, mean |vel|, wrist end-effector error.

Usage:
    .venv/bin/python tools_arm_study_f1.py --bvh downloads/lafan1/dance1_subject3.bvh \
        --start 0 --end 300
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DEFAULT_GMR_ROOT = REPO / "GMR"
URDF = REPO / "F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"

F1_JOINT_NAMES = (
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
)
ARM_JOINTS = [n for n in F1_JOINT_NAMES if any(t in n for t in ("shoulder", "elbow", "wrist"))]
ARM_BODIES = {  # side -> (arm joint names, wrist body, human hand, human arm)
    "left": ([n for n in ARM_JOINTS if n.startswith("left")],
             "left_wrist_roll_link", "LeftHand", "LeftArm"),
    "right": ([n for n in ARM_JOINTS if n.startswith("right")],
              "right_wrist_roll_link", "RightHand", "RightArm"),
}


def urdf_limits():
    tree = ET.parse(URDF)
    out = {}
    for j in tree.getroot().iter("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        lim = j.find("limit")
        out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def apply_variant(retargeter, variant: str):
    """Modify ik_match_tables in place; caller must re-setup configuration."""
    if variant == "v0_baseline" or variant.startswith("v3"):
        return
    soft = {"v1_soft_wrist": {"wrist_roll_link": ("rot", 1.0)},
            "v2_soft_arm": {"wrist_roll_link": ("rot", 1.0),
                            "elbow_pitch_link": ("rot", 3.0),
                            "shoulder_pitch_link": ("rot", 5.0)}}
    table = soft.get(variant, soft["v2_soft_arm"])
    for tbl in (retargeter.ik_match_table1, retargeter.ik_match_table2):
        for frame_name, entry in tbl.items():
            for key, (kind, val) in table.items():
                if frame_name.endswith(key):
                    entry[2 if kind == "rot" else 1] = val


RELOCK_ERRORS: dict = {}
RELOCK_POST_REF: dict = {}  # side -> low-pass posture reference (qpos vector)


def arm_relock(retargeter, qpos, scaled_human, mj, mink,
               posture_cost=2.0, wrist_pos_cost=50.0, reach_scale=0.92,
               lowpass_posture=True, lp_alpha=0.35):
    """Secondary per-arm IK: wrist position task + posture regularizer.

    The wrist target is RE-ANCHORED to the robot's FK shoulder position:
    target = shoulder_fk + clamp(scaled_human_hand - scaled_human_arm,
    reach_scale * arm reach). The whole-body IK often leaves the hand target
    just outside the arm's reachable sphere, which saturates every arm joint
    against its limit; clamping makes the target always reachable. The
    posture term pulls joints toward a low-pass of the solved pose so slow
    joints stay where they were (motion preserved) instead of being pinned
    at the range center or at a limit.
    """
    model = retargeter.model
    qpos = qpos.copy()
    for side, (joints, wrist_body, human_hand, human_arm) in ARM_BODIES.items():
        jids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in joints]
        arm_dofs = {model.jnt_dofadr[j] for j in jids}
        frozen = [d for d in range(model.nv) if d not in arm_dofs]
        cfg = mink.Configuration(model)
        cfg.data.qpos[:] = qpos
        mj.mj_forward(model, cfg.data)

        # F1 arm reach (shoulder->elbow->wrist body distances at this pose)
        shoulder_body = wrist_body.replace("wrist_roll", "shoulder_pitch")
        elbow_body = wrist_body.replace("wrist_roll", "elbow_pitch")
        sp = cfg.data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, shoulder_body)].copy()
        ep = cfg.data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, elbow_body)].copy()
        wp = cfg.data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, wrist_body)].copy()
        reach = np.linalg.norm(ep - sp) + np.linalg.norm(wp - ep)

        # re-anchor + clamp the hand target to the robot's shoulder FK
        offset = np.asarray(scaled_human[human_hand][0]) - np.asarray(scaled_human[human_arm][0])
        L = np.linalg.norm(offset)
        maxL = reach_scale * reach
        if L > maxL:
            offset = offset * (maxL / L)
        target = sp + offset

        # posture target: the whole-body solution itself (motion semantics
        # preserved), with a safety belt pulling saturated joints at least
        # `belt` rad inside their limits
        belt = 0.12
        tgt = qpos.copy()
        for j in jids:
            adr = model.jnt_qposadr[j]
            tgt[adr] = np.clip(qpos[adr], model.jnt_range[j][0] + belt, model.jnt_range[j][1] - belt)
        posture = mink.PostureTask(model, cost=posture_cost)
        posture.set_target(tgt)

        wrist_task = mink.FrameTask(frame_name=wrist_body, frame_type="body",
                                    position_cost=wrist_pos_cost, orientation_cost=0.0,
                                    lm_damping=1)
        wrist_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(np.array([1.0, 0.0, 0.0, 0.0])), target))

        freeze = mink.DofFreezingTask(model, frozen)
        dt = model.opt.timestep
        for _ in range(30):
            vel = mink.solve_ik(cfg, [wrist_task, posture], dt, solver="proxqp",
                                damping=5e-1, limits=[mink.ConfigurationLimit(model)],
                                constraints=[freeze], safety_break=False)
            cfg.integrate_inplace(vel, dt)
        mj.mj_forward(model, cfg.data)
        final_err = float(np.linalg.norm(
            cfg.data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, wrist_body)] - target))
        out = cfg.data.qpos.copy()
        for j in jids:
            qpos[model.jnt_qposadr[j]] = out[model.jnt_qposadr[j]]
        # update the low-pass reference from the solved arm pose
        ref = RELOCK_POST_REF.setdefault(side, qpos.copy())
        for j in jids:
            adr = model.jnt_qposadr[j]
            ref[adr] = (1 - lp_alpha) * ref[adr] + lp_alpha * qpos[adr]
        RELOCK_ERRORS.setdefault(side, []).append(final_err)
    return qpos


def parse_v3(variant: str):
    """v3_c2.0_r0.92_lp0.35 -> dict(posture_cost=2.0, reach_scale=0.92, lp_alpha=0.35)"""
    cfgs = variant.removeprefix("v3").strip("_").split("_")
    out = {}
    for c in cfgs:
        if c.startswith("c"):
            out["posture_cost"] = float(c[1:])
        elif c.startswith("r"):
            out["reach_scale"] = float(c[1:])
        elif c.startswith("lp"):
            out["lp_alpha"] = float(c[2:])
    return out


def score_arm(joints, fps, sat_margin=0.05):
    lim = urdf_limits()
    arm_idx = [F1_JOINT_NAMES.index(n) for n in ARM_JOINTS]
    aj = joints[:, arm_idx]
    lo = np.array([lim[n][0] for n in ARM_JOINTS])
    up = np.array([lim[n][1] for n in ARM_JOINTS])
    sat = ((aj <= lo[None] + sat_margin) | (aj >= up[None] - sat_margin))
    vel = np.abs(np.diff(aj, axis=0)) * fps
    per = {}
    for k, n in enumerate(ARM_JOINTS):
        per[n] = dict(sat=float(sat[:, k].mean() * 100),
                      rom=float((np.percentile(aj[:, k], 95) - np.percentile(aj[:, k], 5))
                                / (up[k] - lo[k]) * 100),
                      vel=float(vel[:, k].mean()))
    return dict(any_sat=float(sat.any(axis=1).mean() * 100), per=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", type=Path, default=REPO / "downloads/lafan1/dance1_subject3.bvh")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=300)
    ap.add_argument("--variants", nargs="+",
                    default=["v0_baseline", "v3_c2.0_r0.92"])
    ap.add_argument("--json_out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(DEFAULT_GMR_ROOT))
    import mujoco as mj
    import mink  # noqa: F401
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_height = load_bvh_file(str(args.bvh), format="lafan1")
    frames = frames[args.start:args.end]
    print(f"[src] {len(frames)} frames from {args.bvh.name}")

    retargeter = GeneralMotionRetargeting(
        actual_human_height=float(human_height), src_human="bvh_lafan1",
        tgt_robot="f1", solver="proxqp", verbose=False, use_velocity_limit=False)

    qpos_adrs, limits = [], []
    for name in F1_JOINT_NAMES:
        jid = mj.mj_name2id(retargeter.model, mj.mjtObj.mjOBJ_JOINT, name)
        qpos_adrs.append(int(retargeter.model.jnt_qposadr[jid]))
        limits.append(retargeter.model.jnt_range[jid].copy())
    qpos_adrs = np.array(qpos_adrs)
    lo = np.array([l[0] for l in limits])
    up = np.array([l[1] for l in limits])

    results = {}
    for variant in args.variants:
        retargeter.setup_retarget_configuration()
        apply_variant(retargeter, variant)
        init = retargeter.configuration.data.qpos.copy()
        for adr, l, u in zip(qpos_adrs, lo, up):
            init[adr] = l + 0.15 * (u - l)
        retargeter.configuration.update(init)
        retargeter.max_iter = 30
        RELOCK_ERRORS.clear()
        RELOCK_POST_REF.clear()
        v3kw = parse_v3(variant) if variant.startswith("v3") else None

        qpos_frames, n_fail = [], 0
        for index, frame in enumerate(frames):
            try:
                qpos = np.asarray(retargeter.retarget(frame), dtype=np.float64)
            except Exception:
                n_fail += 1
                qpos = qpos_frames[-1].copy()
            if v3kw is not None:
                qpos = arm_relock(retargeter, qpos, retargeter.scaled_human_data,
                                  mj, mink, **v3kw)
            qpos[qpos_adrs] = np.clip(qpos[qpos_adrs], lo + 0.01, up - 0.01)
            qpos_frames.append(qpos)
            if index % 100 == 0:
                print(f"[{variant}] frame {index}/{len(frames)}")
        motion = np.stack(qpos_frames)
        joints = motion[:, qpos_adrs]
        sc = score_arm(joints, fps=30)
        sc["ik_failures"] = n_fail
        if v3kw is not None:
            sc["relock_wrist_err_m"] = {s: float(np.mean(v)) for s, v in RELOCK_ERRORS.items()}
        results[variant] = sc
        worst = sorted(((v["sat"], k) for k, v in sc["per"].items() if v["sat"] > 5), reverse=True)[:6]
        rom_dead = [k for k, v in sc["per"].items() if v["rom"] < 2]
        extra = ""
        if "relock_wrist_err_m" in sc:
            extra = " | wrist err " + "/".join(f"{s} {e*1000:.0f}mm" for s, e in sc["relock_wrist_err_m"].items())
        print(f"[{variant}] any_sat {sc['any_sat']:.1f}%  fails {n_fail} | worst: "
              + (", ".join(f"{k.replace('_joint','')} {s:.0f}%" for s, k in worst) or "none")
              + (" | ROM<2%: " + ", ".join(n.replace("_joint", "") for n in rom_dead) if rom_dead else "")
              + extra)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"json -> {args.json_out}")


if __name__ == "__main__":
    main()
