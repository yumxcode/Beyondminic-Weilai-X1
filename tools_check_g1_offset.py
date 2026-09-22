"""Validate rot_offset formula: rot_offset = R_conv * R_body_neutral (quat composition).

Compares computed values against the known-good smplx_to_g1.json config.
"""
import mujoco
import numpy as np
import json

from scipy.spatial.transform import Rotation as R

m = mujoco.MjModel.from_xml_path("GMR/assets/unitree_g1/g1_mocap_29dof.xml")
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)

mujoco.mj_forward(m, d)

R_conv = R.from_quat([0.5, -0.5, -0.5, -0.5], scalar_first=True)
cfg = json.load(open("GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json"))

print(f"{'body':24s} {'config quat':40s} match")
for body, entry in cfg["ik_match_table1"].items():
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        print(f"{body:24s} NOT FOUND")
        continue
    wq = d.xquat[bid]
    comp = (R_conv * R.from_quat(wq, scalar_first=True)).as_quat(scalar_first=True)
    cfgq = np.array(entry[4]) / np.linalg.norm(entry[4])
    dot = abs(np.dot(comp, cfgq))
    print(f"{body:24s} {str(entry[4]):40s} {'OK' if dot > 0.999 else f'DIFF(dot={dot:.3f})'}")
