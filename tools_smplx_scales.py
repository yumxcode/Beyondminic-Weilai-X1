"""Compute SMPL-X canonical segment lengths from the local neutral pkl."""
import numpy as np
import pickle

d = pickle.load(open("GMR/assets/body_models/smplx/SMPLX_NEUTRAL.pkl", "rb"), encoding="latin1")
J = d["J_regressor"].toarray() if hasattr(d["J_regressor"], "toarray") else np.asarray(d["J_regressor"])
v = d["v_template"]
joints = J @ v  # (55, 3), SMPL-X canonical joints (y-up)

# SMPL-X joint indices: 0 pelvis, 2 left_hip, 5 right_hip, 10 left_knee, 11 right_knee,
# 12 left_ankle? Actually use names via smpl convention:
# 0 pelvis, 1 left_hip, 2 right_hip, 4 left_knee, 5 right_knee, 7 left_ankle, 8 right_ankle,
# 12 left_shoulder, 13 right_shoulder, 14 left_elbow, 15 right_elbow, 16 left_wrist, 17 right_wrist,
# 15/16 spine... exact indices below per SMPLX joint_names order:
names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
         "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "left_collar",
         "right_collar", "neck", "head", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye_smplhf", "right_eye_smplhf"]
idx = {n: i for i, n in enumerate(names)}
pelvis = joints[idx["pelvis"]]
lhip, lankle = joints[idx["left_hip"]], joints[idx["left_ankle"]]
lshoulder, lwrist = joints[idx["left_shoulder"]], joints[idx["left_wrist"]]
spine3 = joints[idx["spine3"]]
foot_y = joints[idx["left_foot"]][1]

print(f"template height (y): {joints.max(0)[1] - joints.min(0)[1]:.3f}")
print(f"pelvis y={pelvis[1]:.3f}  spine3 y={spine3[1]:.3f}")
print(f"left hip y={lhip[1]:.3f}  left ankle y={lankle[1]:.3f}  left foot y={foot_y:.3f}")
print(f"left shoulder y={lshoulder[1]:.3f}  wrist y={lwrist[1]:.3f} dy={lshoulder[1]-lwrist[1]:.3f}")
leg = pelvis[1] - lankle[1]
spine = spine3[1] - pelvis[1]
arm = lshoulder[1] - lwrist[1]
print(f"\nsegments: leg(pelvis->ankle)={leg:.4f}  spine(pelvis->spine3)={spine:.4f}  arm(shoulder->wrist,T)={arm:.4f}")

# X1 dims (from tools_measure_scales.py FK):
x1_leg, x1_arm, x1_spine = 0.5726, 0.4215, 0.1560
print(f"\nX1 scales for smplx: legs={x1_leg/leg:.4f} arms={x1_arm/arm:.4f} spine={x1_spine/spine:.4f} pelvis={x1_leg/leg:.4f}")
