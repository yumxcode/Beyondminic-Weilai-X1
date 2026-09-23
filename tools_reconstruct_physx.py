"""Reconstruct the PhysX joint order position-by-position from normalizer stats.

For each policy obs slot i in [0,29), match the (mean, std) pair against every
joint's motion (mean, std); the best match reveals which joint that slot
tracks. Adaptive sampling biases both sides similarly, so this is robust.
"""
import numpy as np
import torch

ckpt = torch.load("trained_models/dance_x1_model_4999.pt", map_location="cpu", weights_only=False)
nst = ckpt["obs_norm_state_dict"]
mean = np.asarray(nst["_mean"]).squeeze()
var = np.asarray(nst["_var"]).squeeze()
std = np.sqrt(var + 1e-6)

npz = np.load("whole_body_tracking/motions/dance_x1_train.npz")
jp = np.asarray(npz["joint_pos"])
jn = [str(s) for s in npz["joint_names"]]

j_mean = jp.mean(axis=0)
j_std = jp.std(axis=0)
# z-score both feature sets so mean and std contribute comparably
def z(v):
    return (v - v.mean()) / (v.std() + 1e-9)

feat_slot = np.stack([z(mean[:29]), z(std[:29])], axis=1)      # (29, 2)
feat_joint = np.stack([z(j_mean), z(j_std)], axis=1)            # (29, 2)

D = np.linalg.norm(feat_slot[:, None, :] - feat_joint[None, :, :], axis=2)  # (29,29)

order = []
used = set()
for i in range(29):
    for j in np.argsort(D[i]):
        if j not in used:
            order.append(jn[j]); used.add(int(j))
            print(f"slot {i:2d} -> {jn[j]:30s} d={D[i,j]:.3f} (next {D[i, np.argsort(D[i])[1]]:.3f})")
            break

print("\nreconstructed PhysX order:")
print(order)
np.save("physx_joint_order.npy", np.array(order))
