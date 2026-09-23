"""Full comparison: normalizer std slots vs npz per-joint vel std (BFS & serial)."""
import numpy as np
import torch

ckpt = torch.load("trained_models/dance_x1_model_4999.pt", map_location="cpu", weights_only=False)
nst = ckpt["obs_norm_state_dict"]
mean = np.asarray(nst["_mean"]).squeeze()
std = np.sqrt(np.asarray(nst["_var"]).squeeze() + 1e-6)

npz = np.load("whole_body_tracking/motions/dance_x1_train.npz")
jn = [str(s) for s in npz["joint_names"]]
jv = np.asarray(npz["joint_vel"])
jp = np.asarray(npz["joint_pos"])

BFS = ["left_hip_pitch_joint", "lumbar_yaw_joint", "right_hip_pitch_joint",
       "left_hip_roll_joint", "lumbar_roll_joint", "right_hip_roll_joint",
       "left_hip_yaw_joint", "lumbar_pitch_joint", "right_hip_yaw_joint",
       "left_knee_pitch_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "right_knee_pitch_joint",
       "left_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "right_ankle_pitch_joint",
       "left_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "right_ankle_roll_joint",
       "left_elbow_pitch_joint", "right_elbow_pitch_joint",
       "left_elbow_yaw_joint", "right_elbow_yaw_joint",
       "left_wrist_pitch_joint", "right_wrist_pitch_joint",
       "left_wrist_roll_joint", "right_wrist_roll_joint"]
idx = {n: i for i, n in enumerate(jn)}

print(f"{'slot':>4s} {'train_std':>10s} {'train_mean':>10s}  {'npzvelstd_bfs':>13s} {'npzposstd_bfs':>13s}  joint(BFS)")
dead = []
for s in range(29):
    n = BFS[s]
    vs = jv[:, idx[n]].std()
    ps = jp[:, idx[n]].std()
    flag = " <-- DEAD" if std[s] < 0.01 else ""
    if std[s] < 0.01:
        dead.append((s, n))
    print(f"{s:4d} {std[s]:10.4f} {mean[s]:10.4f}  {vs:13.4f} {ps:13.4f}  {n}{flag}")
print("\ndead slots:", dead)
