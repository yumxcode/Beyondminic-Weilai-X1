#!/usr/bin/env python3
"""F1 foot sole posture + arm quality diagnostics for retargeted motions.

Quantifies the two user-reported problems on any retargeted F1 motion
(csv from tools_bvh_to_csv_f1.py / tools_amc_csv_f1.py, or npz):

  FOOT (per side):
    - contact frames: lowest sole corner z < --contact_h (near-ground)
    - stance frames: contact runs >= --stance_min_frames (support phase,
      excludes brief toe-off taps)
    - sole pitch (heel-toe):  atan2(heel_z - toe_z, foot_len)
        + = heel up   (tiptoe / plantar flexion)
        - = toe up    (heel-strike flare)
    - sole roll  (med-lat):   atan2(medial_z - lateral_z, foot_wid)
        + = medial edge up (inversion / supination)
        - = lateral edge up (eversion / pronation)
    - flat rate: |pitch|<=p_lim AND |roll|<=r_lim over stance frames
    - ankle_pitch/ankle_roll joint angle distribution over stance frames
  ARM (14 arm joints, URDF limits):
    - limit saturation: frames within --sat_margin of either limit
    - range utilization: (p95-p5)/(upper-lower)
    - smoothness: mean |joint vel|, mean |jerk|

Usage:
    python tools_diag_foot_f1.py --csv whole_body_tracking/motions/csv/dance_f1_train.csv --fps 30
    python tools_diag_foot_f1.py --npz whole_body_tracking/motions/cmu137_24_f1.npz
"""
from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
F1_XML = REPO / "GMR/assets/f1/f1_mocap.xml"
URDF = REPO / "F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"

F1_JOINT_NAMES = [
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
ARM_JOINTS = [n for n in F1_JOINT_NAMES if any(t in n for t in ("shoulder", "elbow", "wrist"))]

# sole corners in ankle_roll frame, order: [heel_lat, toe_lat, heel_med, toe_med]
# left foot: y=-0.0395 lateral(outside), +0.0365 medial(inside)
# right foot: y=-0.0365 medial(inside), +0.0395 lateral(outside)
FOOT_CORNERS = {
    "left":  [(-0.0967, -0.0395, -0.0337), (0.1071, -0.0395, -0.0337),
              (-0.0967,  0.0365, -0.0337), (0.1071,  0.0365, -0.0337)],
    "right": [(-0.0967, -0.0365, -0.0337), (0.1071, -0.0365, -0.0337),
              (-0.0967,  0.0395, -0.0337), (0.1071,  0.0395, -0.0337)],
}
FOOT_LEN = 0.1071 - (-0.0967)   # 0.2038 m heel->toe
FOOT_WID = 0.076                  # m medial<->lateral


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
        return raw[:, :3], raw[:, 3:7], raw[:, 7:], float(args.fps)
    data = np.load(args.npz)
    fps = float(data["fps"])
    body_names = [str(b) for b in data["body_names"]]
    jp = data["joint_pos"]
    base_idx = body_names.index("base_link")
    root_pos = data["body_pos_w"][:, base_idx]
    root_quat = data["body_quat_w"][:, base_idx][:, [1, 2, 3, 0]]  # wxyz->xyzw
    return root_pos, root_quat, jp, fps


def stance_mask(contact: np.ndarray, min_frames: int) -> np.ndarray:
    """Frames belonging to contact runs of length >= min_frames."""
    out = np.zeros_like(contact)
    n = len(contact)
    i = 0
    while i < n:
        if contact[i]:
            j = i
            while j < n and contact[j]:
                j += 1
            if j - i >= min_frames:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--contact_h", type=float, default=0.03,
                    help="lowest sole corner z below this = contact (m)")
    ap.add_argument("--stance_min_frames", type=int, default=5,
                    help="contact run length for stance phase (5 @30fps = 0.17 s)")
    ap.add_argument("--pitch_lim_deg", type=float, default=10.0)
    ap.add_argument("--roll_lim_deg", type=float, default=8.0)
    ap.add_argument("--sat_margin", type=float, default=0.05,
                    help="arm joint within this of a limit = saturated (rad)")
    ap.add_argument("--json_out", type=Path, default=None)
    args = ap.parse_args()
    if not (args.csv or args.npz):
        ap.error("need --csv or --npz")

    root_pos, root_quat, joints, fps = load_motion(args)
    n_frames = len(joints)
    limits = urdf_limits()

    model = mujoco.MjModel.from_xml_path(str(F1_XML))
    data = mujoco.MjData(model)
    jid_of = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in F1_JOINT_NAMES}
    qadr = np.array([model.jnt_qposadr[jid_of[n]] for n in F1_JOINT_NAMES])
    bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll")
           for s in ("left", "right")}

    # per-frame sole corner z (world) and ankle joint values
    corner_z = {s: np.zeros((n_frames, 4)) for s in ("left", "right")}
    for f in range(n_frames):
        q = np.zeros(model.nq)
        mujoco.mj_resetData(model, data)
        q[:] = data.qpos
        q[0:3] = root_pos[f]
        qq = root_quat[f] / np.linalg.norm(root_quat[f])
        q[3:7] = [qq[3], qq[0], qq[1], qq[2]]
        q[qadr] = joints[f]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for s in ("left", "right"):
            R = data.xmat[bid[s]].reshape(3, 3)
            for k, c in enumerate(FOOT_CORNERS[s]):
                p = data.xpos[bid[s]] + R @ np.array(c)
                corner_z[s][f, k] = p[2]

    report = {"frames": n_frames, "fps": fps, "foot": {}}
    print(f"motion: {n_frames} frames @ {fps:.0f} fps")
    for s in ("left", "right"):
        cz = corner_z[s]
        lowest = cz.min(axis=1)
        contact = lowest < args.contact_h
        stance = stance_mask(contact, args.stance_min_frames)
        heel = cz[:, [0, 2]].mean(axis=1)
        toe = cz[:, [1, 3]].mean(axis=1)
        medial = cz[:, [2, 3]].mean(axis=1)
        lateral = cz[:, [0, 1]].mean(axis=1)
        pitch = np.degrees(np.arctan2(heel - toe, FOOT_LEN))   # + = heel up (tiptoe)
        roll = np.degrees(np.arctan2(medial - lateral, FOOT_WID))  # + = inversion

        st = stance if stance.sum() >= 10 else contact  # fall back to all contact
        phase = "stance" if stance.sum() >= 10 else "contact"
        flat = (np.abs(pitch[st]) <= args.pitch_lim_deg) & (np.abs(roll[st]) <= args.roll_lim_deg)

        ap_name = f"{s}_ankle_pitch_joint"
        ar_name = f"{s}_ankle_roll_joint"
        ap_idx = F1_JOINT_NAMES.index(ap_name)
        ar_idx = F1_JOINT_NAMES.index(ar_name)

        rep = {
            "contact_frames": int(contact.sum()),
            f"{phase}_frames": int(st.sum()),
            "sole_pitch_deg": {"mean": float(pitch[st].mean()), "p5": pct(pitch[st], 5),
                               "p50": pct(pitch[st], 50), "p95": pct(pitch[st], 95),
                               "abs_gt_lim_pct": float((np.abs(pitch[st]) > args.pitch_lim_deg).mean() * 100)},
            "sole_roll_deg": {"mean": float(roll[st].mean()), "p5": pct(roll[st], 5),
                              "p50": pct(roll[st], 50), "p95": pct(roll[st], 95),
                              "abs_gt_lim_pct": float((np.abs(roll[st]) > args.roll_lim_deg).mean() * 100)},
            "flat_rate_pct": float(flat.mean() * 100),
            "ankle_pitch_joint_stance_deg": {"mean": float(np.degrees(joints[st, ap_idx].mean())),
                                             "min": float(np.degrees(joints[st, ap_idx].min())),
                                             "max": float(np.degrees(joints[st, ap_idx].max()))},
            "ankle_roll_joint_stance_deg": {"mean": float(np.degrees(joints[st, ar_idx].mean())),
                                            "min": float(np.degrees(joints[st, ar_idx].min())),
                                            "max": float(np.degrees(joints[st, ar_idx].max()))},
        }
        report["foot"][s] = rep
        print(f"[{s} foot] {phase} frames {int(st.sum())}/{n_frames} | "
              f"pitch mean {rep['sole_pitch_deg']['mean']:+.1f}° p95 {rep['sole_pitch_deg']['p95']:+.1f}° "
              f"(|>{args.pitch_lim_deg:.0f}°: {rep['sole_pitch_deg']['abs_gt_lim_pct']:.1f}%) | "
              f"roll mean {rep['sole_roll_deg']['mean']:+.1f}° p95 {rep['sole_roll_deg']['p95']:+.1f}° "
              f"(|>{args.roll_lim_deg:.0f}°: {rep['sole_roll_deg']['abs_gt_lim_pct']:.1f}%) | "
              f"flat {rep['flat_rate_pct']:.1f}% | "
              f"ankle p/r {rep['ankle_pitch_joint_stance_deg']['mean']:+.1f}/"
              f"{rep['ankle_roll_joint_stance_deg']['mean']:+.1f}°")

    # ---- arm diagnostics ------------------------------------------------
    dt = 1.0 / fps
    arm_idx = [F1_JOINT_NAMES.index(n) for n in ARM_JOINTS]
    aj = joints[:, arm_idx]
    lo = np.array([limits[n][0] for n in ARM_JOINTS])
    up = np.array([limits[n][1] for n in ARM_JOINTS])
    sat = ((aj <= lo[None, :] + args.sat_margin) | (aj >= up[None, :] - args.sat_margin))
    vel = np.abs(np.diff(aj, axis=0)) / dt
    acc = np.abs(np.diff(aj, n=2, axis=0)) / dt**2

    per_joint = {}
    for k, name in enumerate(ARM_JOINTS):
        rng = up[k] - lo[k]
        util = (pct(aj[:, k], 95) - pct(aj[:, k], 5)) / rng
        per_joint[name] = {
            "sat_frames": int(sat[:, k].sum()),
            "sat_pct": float(sat[:, k].mean() * 100),
            "range_util_pct": float(util * 100),
            "mean_abs_vel_rad_s": float(vel[:, k].mean()) if len(vel) else 0.0,
            "mean_abs_acc_rad_s2": float(acc[:, k].mean()) if len(acc) else 0.0,
        }
    arm_any_sat = sat.any(axis=1)
    report["arm"] = {
        "sat_margin_rad": args.sat_margin,
        "any_arm_joint_saturated_pct": float(arm_any_sat.mean() * 100),
        "per_joint": per_joint,
    }
    print(f"[arm] any-joint sat(±{args.sat_margin} rad) {arm_any_sat.mean()*100:.1f}% | "
          + " | ".join(f"{n.replace('_joint','').replace('left_','L').replace('right_','R')}:"
                      f"{per_joint[n]['sat_pct']:.0f}%sat/{per_joint[n]['range_util_pct']:.0f}%rom"
                      for n in ARM_JOINTS if per_joint[n]["sat_pct"] > 0 or per_joint[n]["range_util_pct"] < 15))

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"json -> {args.json_out}")


if __name__ == "__main__":
    main()
