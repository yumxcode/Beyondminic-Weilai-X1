#!/usr/bin/env python3
"""Generate GMR ik_configs for the F1 robot (29-DoF, same topology as X1).

Recipe (from tools_build_x1_ikcfg.py, validated for X1):
    rot_offset(F1 body) = conv(T1-equivalent body) * R_f1_body_neutral

Upper body (lumbar + arms): F1 and X1 share byte-identical URDF joint
definitions and frames, so the neutral orientations are identical and the
X1 entries — which were additionally hand-calibrated for arm rot_offsets and
foot pos weights (tools_calibrate_x1.py, wrist err 0.37->0.17) — are reused
VERBATIM from the committed bvh_lafan1_to_xyber_x1.json.

Legs: F1 frames differ (no rpy rotations in the F1 leg chain) -> rot_offsets
recomputed from the T1 conversion * F1 neutral FK.

human_scale_table: recomputed from F1 FK segment lengths (leg 0.5726/0.831,
arm 0.4215/0.553, spine measured) — values expected to match X1's closely.
Foot pos_off: aligns the F1 ankle height above sole (0.0337 m from vendor
collision mesh AABB) with the scaled human ankle height.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent
GMR = REPO / "GMR"
F1_XML = GMR / "assets/f1/f1_mocap.xml"
X1_CFG = GMR / "general_motion_retargeting/ik_configs/bvh_lafan1_to_xyber_x1.json"
X1_CFG_SMPLX = GMR / "general_motion_retargeting/ik_configs/smplx_to_xyber_x1.json"

LAFAN_MAP = {
    "base_link": "Hips",
    "left_hip_pitch_link": "LeftUpLeg",
    "left_knee_pitch_link": "LeftLeg",
    "left_ankle_roll_link": "LeftFootMod",
    "right_hip_pitch_link": "RightUpLeg",
    "right_knee_pitch_link": "RightLeg",
    "right_ankle_roll_link": "RightFootMod",
    "lumbar_pitch_link": "Spine2",
    "left_shoulder_pitch_link": "LeftArm",
    "left_elbow_pitch_link": "LeftForeArm",
    "left_wrist_roll_link": "LeftHand",
    "right_shoulder_pitch_link": "RightArm",
    "right_elbow_pitch_link": "RightForeArm",
    "right_wrist_roll_link": "RightHand",
}

# F1 body names (legs lack the _link suffix)
F1_BODY = {k: (k if k == "base_link" else
               (k if k.endswith("_link") and k.split("_")[0] in ("lumbar",) else k))
           for k in LAFAN_MAP}


def f1_body_name(name: str) -> str:
    if name == "base_link" or name.startswith("lumbar") or "shoulder" in name \
            or "elbow" in name or "wrist" in name:
        return name
    return name.removesuffix("_link")  # legs: left_hip_pitch_link -> left_hip_pitch


def neutral_F1() -> dict[str, R]:
    m = mujoco.MjModel.from_xml_path(str(F1_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    out = {}
    for name in LAFAN_MAP:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f1_body_name(name))
        assert bid >= 0, f"body {name} missing in F1 mocap xml"
        out[name] = R.from_quat(d.xquat[bid], scalar_first=True)
    return out


T1_MAP_LAFAN = {
    "base_link": "Waist", "left_hip_pitch_link": "Hip_Yaw_Left",
    "left_knee_pitch_link": "Shank_Left", "left_ankle_roll_link": "left_foot_link",
    "right_hip_pitch_link": "Hip_Yaw_Right", "right_knee_pitch_link": "Shank_Right",
    "right_ankle_roll_link": "right_foot_link",
    "lumbar_pitch_link": "Trunk",
    "left_shoulder_pitch_link": "AL2", "left_elbow_pitch_link": "AL4",
    "left_wrist_roll_link": "AL5",
    "right_shoulder_pitch_link": "AR2", "right_elbow_pitch_link": "AR4",
    "right_wrist_roll_link": "AR5",
}
T1_MAP_SMPLX = dict(T1_MAP_LAFAN,
                    left_wrist_roll_link="left_hand_link",
                    right_wrist_roll_link="right_hand_link")


def t1_conv(config_file: str) -> dict[str, R]:
    """T1-equivalent conversion rotations (identity-oriented T1 bodies)."""
    cfg = json.load(open(GMR / "general_motion_retargeting/ik_configs" / config_file))
    T1_MAP = T1_MAP_SMPLX if "smplx" in config_file else T1_MAP_LAFAN
    conv = {}
    for body, t1_body in T1_MAP.items():
        entry = cfg["ik_match_table1"][t1_body]
        q = np.array(entry[4], dtype=float)
        conv[body] = R.from_quat(q / np.linalg.norm(q), scalar_first=True)
    return conv


def main() -> None:
    neu = neutral_F1()
    x1_cfg = json.load(open(X1_CFG))
    x1_smplx = json.load(open(X1_CFG_SMPLX))

    # Per-body neutral comparison vs X1: X1's committed entries carry hand
    # calibration (tools_calibrate_x1.py) — reuse ONLY where the F1 neutral
    # frame actually matches (wrists and right-arm frames differ!).
    x1m = mujoco.MjModel.from_xml_path(str(GMR / "assets/xyber_x1/xyber_x1_mocap.xml"))
    x1d = mujoco.MjData(x1m)
    mujoco.mj_resetData(x1m, x1d)
    mujoco.mj_forward(x1m, x1d)
    reuse = {}
    for name in LAFAN_MAP:
        if "hip" in name or "knee" in name or "ankle" in name:
            reuse[name] = False
            continue
        bid = mujoco.mj_name2id(x1m, mujoco.mjtObj.mjOBJ_BODY, name)
        rx = R.from_quat(x1d.xquat[bid], scalar_first=True)
        diff_deg = (rx.inv() * neu[name]).magnitude() * 180 / np.pi
        reuse[name] = diff_deg < 0.1
        print(f"  {name:30s} X1-neutral diff {diff_deg:8.4f} deg -> "
              f"{'REUSE X1 calibrated' if reuse[name] else 'recompute from T1 conv'}")

    conv_bvh = t1_conv("bvh_lafan1_to_t1_29dof.json")
    conv_smplx = t1_conv("smplx_to_t1_29dof.json")

    # leg scales from F1 FK
    m = mujoco.MjModel.from_xml_path(str(F1_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    def pos(name):
        return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f1_body_name(name))]

    leg_len = np.linalg.norm(pos("left_ankle_roll_link") - pos("left_hip_pitch_link"))
    arm_len = np.linalg.norm(pos("left_wrist_roll_link") - pos("left_shoulder_pitch_link"))
    spine_len = np.linalg.norm(pos("lumbar_pitch_link") - pos("base_link"))
    # LAFAN human segments (from X1 builder comments): leg 0.831, arm 0.553, spine 0.317
    # SMPL-X: leg 0.8746, arm 0.5105, spine 0.2940
    print(f"F1 FK: leg {leg_len:.4f} arm {arm_len:.4f} spine {spine_len:.4f}")

    # F1 ankle height above sole: vendor collision AABB z_bottom = -0.0337
    ANKLE_H = 0.0337

    for src, conv, human_seg, x1_src_cfg, fname in (
        ("bvh_lafan1", conv_bvh, (0.831, 0.553, 0.317), x1_cfg, "bvh_lafan1_to_f1.json"),
        ("smplx", conv_smplx, (0.8746, 0.5105, 0.2940), x1_smplx, "smplx_to_f1.json"),
    ):
        leg_s = leg_len / human_seg[0]
        arm_s = arm_len / human_seg[1]
        spine_s = spine_len / human_seg[2]
        scale = {}
        for k in x1_src_cfg["human_scale_table"]:
            if any(t in k for t in ("Leg", "Foot", "Hips")):
                scale[k] = round(leg_s, 3)
            elif any(t in k for t in ("Arm", "Hand")):
                scale[k] = round(arm_s, 3)
            else:
                scale[k] = round(spine_s, 3)

        def rot_offset(body):
            if reuse.get(body):
                return np.array(x1_src_cfg["ik_match_table1"][body][4], dtype=float)
            r = conv[body] * neu[body]
            return r.as_quat(scalar_first=True)

        foot_off = ANKLE_H - leg_s * (0.074 if src == "bvh_lafan1" else 0.058)
        print(f"{src}: leg_s {leg_s:.3f} arm_s {arm_s:.3f} spine_s {spine_s:.3f} foot_off_z {foot_off:+.4f}")

        def tables(foot_pos_w1, foot_pos_w2):
            def entry(body, human, pos_w, rot_w, off):
                return [human, pos_w, rot_w, [float(round(v, 4)) for v in off],
                        [float(round(v, 10)) for v in rot_offset(body)]]
            t1, t2 = {}, {}
            t1["base_link"] = entry("base_link", LAFAN_MAP["base_link"], 100, 10, [0, 0, 0])
            t2["base_link"] = entry("base_link", LAFAN_MAP["base_link"], 100, 5, [0, 0, 0])
            for key, human in LAFAN_MAP.items():
                if key == "base_link":
                    continue
                t1rw = 100 if key == "lumbar_pitch_link" and src == "smplx" else 10
                t1[f1_body_name(key)] = entry(key, human, 0, t1rw, [0, 0, 0])
                t2[f1_body_name(key)] = entry(key, human, 10, 10, [0, 0, 0])
            for side in ("left", "right"):
                key = f"{side}_ankle_roll_link"
                offz = foot_off  # F1 foot local +z is up at neutral (both sides)
                t1[f1_body_name(key)] = entry(key, LAFAN_MAP[key], foot_pos_w1, 10, [0, 0, offz])
                t2[f1_body_name(key)] = entry(key, LAFAN_MAP[key], foot_pos_w2, 10, [0, 0, offz])
            return t1, t2

        # X1 calibrated foot pos weight 500 (table1) / 50 (table2)
        t1, t2 = tables(500, 50)
        cfg = {
            "robot_root_name": "base_link",
            "human_root_name": x1_src_cfg["human_root_name"],
            "ground_height": 0.0,
            "human_height_assumption": 1.8,
            "use_ik_match_table1": True,
            "use_ik_match_table2": True,
            "human_scale_table": scale,
            "ik_match_table1": t1,
            "ik_match_table2": t2,
        }
        out = GMR / "general_motion_retargeting/ik_configs" / fname
        out.write_text(json.dumps(cfg, indent=4))
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
