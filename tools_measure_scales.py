"""Measure X1 segment lengths (FK, straight neutral) and LAFAN human segment
lengths, then compute proper human_scale_table values for smplx_to_xyber_x1
and bvh_lafan1_to_xyber_x1."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "GMR"))

import mujoco

m = mujoco.MjModel.from_xml_path(str(REPO / "GMR/assets/xyber_x1/xyber_x1_mocap.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)  # all joints at 0, root at qpos0
mujoco.mj_forward(m, d)


def pos(name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return d.xpos[bid].copy()


def jpos(name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    adr = m.jnt_qposadr[jid]
    # joint anchor world position: xpos of the body owning the joint
    bid = m.jnt_bodyid[jid]
    return d.xpos[bid].copy()


base = pos("base_link")
lumbar = jpos("lumbar_pitch_joint")
lhip = jpos("left_hip_pitch_joint")
lankle = jpos("left_ankle_pitch_joint")
lsole = pos("left_ankle_roll_link")
lshoulder = jpos("left_shoulder_pitch_joint")
lwrist = jpos("left_wrist_roll_joint")

print("X1 FK neutral (root z=%.3f):" % base[2])
print(f"  base->lumbar_pitch  dz = {lumbar[2]-base[2]:.4f}")
print(f"  hip->ankle          dz = {lankle[2]-lhip[2]:.4f}  |d|={np.linalg.norm(lankle-lhip):.4f}")
print(f"  ankle->foot_origin  dz = {lsole[2]-lankle[2]:.4f}")
print(f"  shoulder->wrist     |d| = {np.linalg.norm(lwrist-lshoulder):.4f}  dy={lwrist[1]-lshoulder[1]:.4f}")
print(f"  base->shoulder      dz = {lshoulder[2]-base[2]:.4f} dy={lshoulder[1]:.4f}")
print(f"  hip dy (from center) = {lhip[1]:.4f}")

# LAFAN standing dims (frame 0)
from general_motion_retargeting.utils.lafan1 import load_bvh_file

frames, _ = load_bvh_file(str(REPO / "downloads/lafan1/dance1_subject2.bvh"), format="lafan1")
f0 = frames[0]
hz = lambda k: f0[k][0][2]
print("\nLAFAN frame0:")
print(f"  Hips z={hz('Hips'):.3f} Spine2 z={hz('Spine2'):.3f}")
print(f"  LeftUpLeg z={hz('LeftUpLeg'):.3f} LeftFoot z={hz('LeftFootMod'):.3f}")
print(f"  LeftArm z={hz('LeftArm'):.3f} LeftHand z={hz('LeftHand'):.3f}")
hip_to_ankle_h = hz("LeftUpLeg") - hz("LeftFootMod")
arm_h = np.linalg.norm(f0["LeftHand"][0] - f0["LeftArm"][0])
spine_h = hz("Spine2") - hz("Hips")
print(f"  human hip->ankle dz={hip_to_ankle_h:.3f}  shoulder->wrist |d|={arm_h:.3f}  hips->spine2 dz={spine_h:.3f}")

x1_leg = lankle[2] - lhip[2]
x1_arm = np.linalg.norm(lwrist - lshoulder)
x1_spine = lumbar[2] - base[2]
print("\nsuggested scales:")
print(f"  legs   = {x1_leg/hip_to_ankle_h:.4f}")
print(f"  arms   = {x1_arm/arm_h:.4f}")
print(f"  spine  = {x1_spine/spine_h:.4f}")
print(f"  pelvis(root height) = standing root z / Hips z = {0.61/hz('Hips'):.4f}")
