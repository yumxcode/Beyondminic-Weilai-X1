#!/usr/bin/env python3
"""Generate GMR ik_configs for the Xyber X1 robot.

Recipe (validated against smplx_to_g1.json, smplx_to_t1_29dof.json and
bvh_lafan1_to_t1_29dof.json):

    rot_offset(X1 body) = rot_offset(T1-equivalent body) * R_x1_body_neutral

where T1 bodies are identity-oriented at their neutral configuration, so the
T1 config values are exactly the source-skeleton conversion rotations, and
R_x1_body_neutral is the X1 body world orientation at qpos0 (from MuJoCo FK).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent
GMR = REPO / "GMR"
X1_XML = GMR / "assets/xyber_x1/xyber_x1_mocap.xml"

# X1 body -> T1-equivalent body (same 29-DoF topology) -> source human keypoint.
# SMPL-X mapping (from smplx_to_t1_29dof.json semantics).
SMPLX_MAP = {
    "base_link": ("Waist", "pelvis"),
    "left_hip_pitch_link": ("Hip_Yaw_Left", "left_hip"),
    "left_knee_pitch_link": ("Shank_Left", "left_knee"),
    "left_ankle_roll_link": ("left_foot_link", "left_foot"),
    "right_hip_pitch_link": ("Hip_Yaw_Right", "right_hip"),
    "right_knee_pitch_link": ("Shank_Right", "right_knee"),
    "right_ankle_roll_link": ("right_foot_link", "right_foot"),
    "lumbar_pitch_link": ("Trunk", "spine3"),
    "left_shoulder_pitch_link": ("AL2", "left_shoulder"),
    "left_elbow_pitch_link": ("AL4", "left_elbow"),
    "left_wrist_roll_link": ("left_hand_link", "left_wrist"),
    "right_shoulder_pitch_link": ("AR2", "right_shoulder"),
    "right_elbow_pitch_link": ("AR4", "right_elbow"),
    "right_wrist_roll_link": ("right_hand_link", "right_wrist"),
}

LAFAN_MAP = {
    "base_link": ("Waist", "Hips"),
    "left_hip_pitch_link": ("Hip_Yaw_Left", "LeftUpLeg"),
    "left_knee_pitch_link": ("Shank_Left", "LeftLeg"),
    "left_ankle_roll_link": ("left_foot_link", "LeftFootMod"),
    "right_hip_pitch_link": ("Hip_Yaw_Right", "RightUpLeg"),
    "right_knee_pitch_link": ("Shank_Right", "RightLeg"),
    "right_ankle_roll_link": ("right_foot_link", "RightFootMod"),
    "lumbar_pitch_link": ("Trunk", "Spine2"),
    "left_shoulder_pitch_link": ("AL2", "LeftArm"),
    "left_elbow_pitch_link": ("AL4", "LeftForeArm"),
    "left_wrist_roll_link": ("AL5", "LeftHand"),
    "right_shoulder_pitch_link": ("AR2", "RightArm"),
    "right_elbow_pitch_link": ("AR4", "RightForeArm"),
    "right_wrist_roll_link": ("AR5", "RightHand"),
}


def t1_conversion(config_file: str) -> dict[str, np.ndarray]:
    cfg = json.load(open(GMR / "general_motion_retargeting/ik_configs" / config_file))
    conv = {}
    for x1_body, (t1_body, _human) in SMPLX_MAP.items():
        entry = cfg["ik_match_table1"][t1_body]
        q = np.array(entry[4], dtype=float)
        conv[x1_body] = R.from_quat(q / np.linalg.norm(q), scalar_first=True)
    return conv


def build(x1_neutral: dict[str, R], src: str) -> dict:
    if src == "smplx":
        conv = t1_conversion("smplx_to_t1_29dof.json")
        mapping = SMPLX_MAP
        human_root = "pelvis"
        # scales measured from FK: X1 segments / SMPL-X template segments
        # (leg 0.5726/0.8746, arm 0.4215/0.5105, spine 0.1560/0.2940)
        scale = {
            "pelvis": 0.6547, "spine3": 0.5307,
            "left_hip": 0.6547, "right_hip": 0.6547, "left_knee": 0.6547, "right_knee": 0.6547,
            "left_foot": 0.6547, "right_foot": 0.6547,
            "left_shoulder": 0.8257, "right_shoulder": 0.8257, "left_elbow": 0.8257, "right_elbow": 0.8257,
            "left_wrist": 0.8257, "right_wrist": 0.8257,
        }
    else:
        conv = {}
        lafan_cfg = json.load(open(GMR / "general_motion_retargeting/ik_configs/bvh_lafan1_to_t1_29dof.json"))
        for x1_body, (t1_body, _human) in LAFAN_MAP.items():
            entry = lafan_cfg["ik_match_table1"][t1_body]
            q = np.array(entry[4], dtype=float)
            conv[x1_body] = R.from_quat(q / np.linalg.norm(q), scalar_first=True)
        mapping = LAFAN_MAP
        human_root = "Hips"
        # scales measured from FK vs LAFAN dance1_subject2 frame 0
        # (leg 0.5726/0.831, arm 0.4215/0.553, spine 0.1560/0.317)
        scale = {
            "Hips": 0.689, "Spine2": 0.493,
            "LeftUpLeg": 0.689, "RightUpLeg": 0.689, "LeftLeg": 0.689, "RightLeg": 0.689,
            "LeftFootMod": 0.689, "RightFootMod": 0.689,
            "LeftArm": 0.762, "RightArm": 0.762, "LeftForeArm": 0.762, "RightForeArm": 0.762,
            "LeftHand": 0.762, "RightHand": 0.762,
        }

    def entry(x1_body: str, pos_w: int, rot_w: int, pos_off: list[float]) -> list:
        t1_body, human = mapping[x1_body]
        rot = conv[x1_body] * x1_neutral[x1_body]
        quat = rot.as_quat(scalar_first=True)
        return [human, pos_w, rot_w, pos_off, [float(round(v, 10)) for v in quat]]

    # Foot pos_offsets align the X1 ankle height above the sole (0.061 m,
    # from FK: ankle 0.041 m above foot corners + 0.02 m sphere radius) with
    # the scaled human ankle height.  The offset is along the foot local
    # y-axis, which is vertical at the neutral pose (+y left, -y right).
    # SMPL-X: human ankle 0.058 above sole -> 0.6547*0.058 = 0.038.
    # LAFAN1: human ankle 0.074 above sole -> 0.689*0.074 = 0.051.
    if src == "smplx":
        FOOT_OFF = 0.061 - 0.6547 * 0.058  # 0.023
    else:
        FOOT_OFF = 0.061 - 0.689 * 0.074  # 0.010

    table1 = {}
    # root: strong position
    table1["base_link"] = entry("base_link", 100, 10, [0.0, 0.0, 0.0])
    # torso (anchor): strong rotation in table1 like T1 (spine3 rot_w=100 for smplx)
    torso_rot_w = 100 if src == "smplx" else 10
    table1["lumbar_pitch_link"] = entry("lumbar_pitch_link", 0, torso_rot_w, [0.0, 0.0, 0.0])
    for side in ("left", "right"):
        table1[f"{side}_hip_pitch_link"] = entry(f"{side}_hip_pitch_link", 0, 10, [0.0, 0.0, 0.0])
        table1[f"{side}_knee_pitch_link"] = entry(f"{side}_knee_pitch_link", 0, 10, [0.0, 0.0, 0.0])
        foot_y = FOOT_OFF if side == "left" else -FOOT_OFF
        table1[f"{side}_ankle_roll_link"] = entry(f"{side}_ankle_roll_link", 50, 10, [0.0, foot_y, 0.0])
        table1[f"{side}_shoulder_pitch_link"] = entry(f"{side}_shoulder_pitch_link", 0, 10, [0.0, 0.0, 0.0])
        table1[f"{side}_elbow_pitch_link"] = entry(f"{side}_elbow_pitch_link", 0, 10, [0.0, 0.0, 0.0])
        table1[f"{side}_wrist_roll_link"] = entry(f"{side}_wrist_roll_link", 0, 10, [0.0, 0.0, 0.0])

    # table2: refine everything with balanced weights (mirrors T1 structure)
    table2 = {}
    table2["base_link"] = entry("base_link", 100, 5, [0.0, 0.0, 0.0])
    table2["lumbar_pitch_link"] = entry("lumbar_pitch_link", 0, 10, [0.0, 0.0, 0.0])
    for side in ("left", "right"):
        table2[f"{side}_hip_pitch_link"] = entry(f"{side}_hip_pitch_link", 10, 5, [0.0, 0.0, 0.0])
        table2[f"{side}_knee_pitch_link"] = entry(f"{side}_knee_pitch_link", 10, 5, [0.0, 0.0, 0.0])
        foot_y = FOOT_OFF if side == "left" else -FOOT_OFF
        table2[f"{side}_ankle_roll_link"] = entry(f"{side}_ankle_roll_link", 50, 10, [0.0, foot_y, 0.0])
        table2[f"{side}_shoulder_pitch_link"] = entry(f"{side}_shoulder_pitch_link", 10, 5, [0.0, 0.0, 0.0])
        table2[f"{side}_elbow_pitch_link"] = entry(f"{side}_elbow_pitch_link", 10, 5, [0.0, 0.0, 0.0])
        table2[f"{side}_wrist_roll_link"] = entry(f"{side}_wrist_roll_link", 10, 5, [0.0, 0.0, 0.0])

    return {
        "robot_root_name": "base_link",
        "human_root_name": human_root,
        "ground_height": 0.0,
        "human_height_assumption": 1.8,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": scale,
        "ik_match_table1": table1,
        "ik_match_table2": table2,
    }


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(X1_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    x1_neutral = {}
    for body in SMPLX_MAP:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        assert bid >= 0, f"body {body} missing in X1 mocap xml"
        x1_neutral[body] = R.from_quat(d.xquat[bid], scalar_first=True)

    out_dir = GMR / "general_motion_retargeting/ik_configs"
    for src, fname in (("smplx", "smplx_to_xyber_x1.json"), ("bvh_lafan1", "bvh_lafan1_to_xyber_x1.json")):
        cfg = build(x1_neutral, src)
        with open(out_dir / fname, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"wrote {fname}")

    print("\nneutral orientations (deg from identity):")
    for body, rot in x1_neutral.items():
        deg = rot.magnitude() * 180 / np.pi
        print(f"  {body:28s} {deg:6.1f}")


if __name__ == "__main__":
    main()
