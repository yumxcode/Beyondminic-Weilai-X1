"""Identify PhysX joint order from checkpoint normalizer (fixed BFS impl)."""
import sys

import numpy as np
import torch

print("[1] loading ckpt...", flush=True)
ckpt = torch.load("trained_models/dance_x1_model_4999.pt", map_location="cpu", weights_only=False)
mean = np.asarray(ckpt["obs_norm_state_dict"]["_mean"]).squeeze()
ref_mean_physx = mean[0:29]
print("[1] ok", ref_mean_physx.shape, flush=True)

print("[2] loading npz...", flush=True)
npz = np.load("whole_body_tracking/motions/dance_x1_train.npz")
jp = np.asarray(npz["joint_pos"])
jn = [str(s) for s in npz["joint_names"]]
SERIAL = list(jn)
print("[2] ok", jp.shape, flush=True)

print("[3] BFS...", flush=True)
# explicit merged tree (fixed links removed) — hand-derived from the URDF:
# base -> [lumbar_yaw, left_hip_pitch, right_hip_pitch]
# lumbar_yaw -> lumbar_roll -> lumbar_pitch -> [L/R shoulder_pitch]
# shoulder_pitch -> shoulder_roll -> shoulder_yaw -> elbow_pitch -> elbow_yaw -> wrist_pitch -> wrist_roll
# hip_pitch -> hip_roll -> hip_yaw -> knee_pitch -> ankle_pitch -> ankle_roll
def chain(prefix, names):
    return [f"{prefix}_{n}_joint" for n in names]

arm = chain("left", ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"])
arm_r = [s.replace("left_", "right_") for s in arm]
leg = chain("left", ["hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll"])
leg_r = [s.replace("left_", "right_") for s in leg]

BFS = (
    ["lumbar_yaw_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
     "lumbar_roll_joint", "left_hip_roll_joint", "right_hip_roll_joint",
     "lumbar_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint"]
)
# level 4: shoulder_pitch L/R, knee L/R
BFS += ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_knee_pitch_joint", "right_knee_pitch_joint"]
# level 5: shoulder_roll L/R, ankle_pitch L/R
BFS += ["left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint"]
# level 6: shoulder_yaw L/R, ankle_roll L/R
BFS += ["left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"]
# level 7: elbow_pitch
BFS += ["left_elbow_pitch_joint", "right_elbow_pitch_joint"]
# level 8: elbow_yaw
BFS += ["left_elbow_yaw_joint", "right_elbow_yaw_joint"]
# level 9: wrist_pitch
BFS += ["left_wrist_pitch_joint", "right_wrist_pitch_joint"]
# level 10: wrist_roll
BFS += ["left_wrist_roll_joint", "right_wrist_roll_joint"]
print("[3] BFS joints:", len(BFS), flush=True)
assert sorted(BFS) == sorted(SERIAL), set(BFS) ^ set(SERIAL)

avg_by_name = {jn[i]: float(jp[:, i].mean()) for i in range(29)}

def score(order):
    vec = np.array([avg_by_name[n] for n in order])
    return float(np.corrcoef(vec, ref_mean_physx)[0, 1]), float(np.abs(vec - ref_mean_physx).mean())

for name, order in (("URDF_SERIAL", SERIAL), ("BFS", BFS)):
    c, mae = score(order)
    print(f"{name:12s} corr={c:+.4f} mae={mae:.4f}", flush=True)

print("\nref_mean_physx[:10]:", np.round(ref_mean_physx[:10], 3))
print("serial avg[:10]:     ", np.round([avg_by_name[n] for n in SERIAL[:10]], 3))
print("bfs avg[:10]:        ", np.round([avg_by_name[n] for n in BFS[:10]], 3))
