#!/usr/bin/env python3
"""
Standalone MuJoCo playback for BeyondMimic RSL-RL motion tracking policy.

Loads a .pt checkpoint (RSL-RL ActorCritic) + .npz reference motion,
reconstructs the observation vector, runs policy inference, and applies
PD control in MuJoCo — no Isaac Sim / ROS required.

Requirements:
    pip install torch mujoco numpy scipy

Usage:
    python scripts/play_mujoco.py \
        --checkpoint trained_models/dance_zui_model_4999.pt \
        --motion motions/dance_zui.npz \
        --mjcf source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml

    Optional:
        --speed 0.5        # slow motion (0.5x)
        --no_normalization # disable obs normalization (debug)
        --seed 42          # fixed seed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import mujoco
import mujoco.viewer

# ─────────────────────────────────────────────────────────────
#  G1 robot parameters (mirrors g1.py, computed at import time)
# ─────────────────────────────────────────────────────────────

_NATURAL_FREQ = 10.0 * 2.0 * np.pi   # 62.832 rad/s
_DAMPING_RATIO = 2.0
_W2 = _NATURAL_FREQ ** 2             # 3947.84

_ARM_5020 = 0.003609725
_ARM_7520_14 = 0.01017752004
_ARM_7520_22 = 0.025101925
_ARM_4010 = 0.00425

_KP_5020 = _ARM_5020 * _W2           # 14.25
_KP_7520_14 = _ARM_7520_14 * _W2     # 40.19
_KP_7520_22 = _ARM_7520_22 * _W2     # 99.09
_KP_4010 = _ARM_4010 * _W2           # 16.78

_KD_5020 = 2.0 * _DAMPING_RATIO * _ARM_5020 * _NATURAL_FREQ    # 0.907
_KD_7520_14 = 2.0 * _DAMPING_RATIO * _ARM_7520_14 * _NATURAL_FREQ  # 2.558
_KD_7520_22 = 2.0 * _DAMPING_RATIO * _ARM_7520_22 * _NATURAL_FREQ  # 6.308
_KD_4010 = 2.0 * _DAMPING_RATIO * _ARM_4010 * _NATURAL_FREQ       # 1.068

# Per-joint regex → (effort_limit, kp, kd)
# Matches g1.py ImplicitActuatorCfg definitions
_JOINT_PARAMS = {
    # legs
    ".*_hip_pitch_joint":  (88.0, _KP_7520_14, _KD_7520_14),
    ".*_hip_roll_joint":   (139.0, _KP_7520_22, _KD_7520_22),
    ".*_hip_yaw_joint":    (88.0, _KP_7520_14, _KD_7520_14),
    ".*_knee_joint":       (139.0, _KP_7520_22, _KD_7520_22),
    # feet (2x scale)
    ".*_ankle_pitch_joint": (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    ".*_ankle_roll_joint":  (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    # waist
    "waist_roll_joint":   (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    "waist_pitch_joint":  (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    "waist_yaw_joint":    (88.0, _KP_7520_14, _KD_7520_14),
    # arms
    ".*_shoulder_pitch_joint": (25.0, _KP_5020, _KD_5020),
    ".*_shoulder_roll_joint":  (25.0, _KP_5020, _KD_5020),
    ".*_shoulder_yaw_joint":   (25.0, _KP_5020, _KD_5020),
    ".*_elbow_joint":          (25.0, _KP_5020, _KD_5020),
    ".*_wrist_roll_joint":     (25.0, _KP_5020, _KD_5020),
    ".*_wrist_pitch_joint":    (5.0, _KP_4010, _KD_4010),
    ".*_wrist_yaw_joint":      (5.0, _KP_4010, _KD_4010),
}

# Default joint positions (from G1_CYLINDER_CFG.init_state)
_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

INIT_HEIGHT = 0.76  # G1_CYLINDER_CFG.init_state.pos[2]

# Sim params (from tracking_env_cfg.py)
SIM_DT = 0.005       # 200 Hz physics
DECIMATION = 4       # policy at 50 Hz
POLICY_DT = SIM_DT * DECIMATION  # 0.02s

ACTOR_HIDDEN_DIMS = [512, 256, 128]
ACTIVATION = "elu"


# ─────────────────────────────────────────────────────────────
#  Helpers: regex matching for joint params
# ─────────────────────────────────────────────────────────────

import re as _re


def _match_joint_params(joint_name: str) -> tuple[float, float, float]:
    """Return (effort_limit, kp, kd) for a joint name."""
    for pattern, params in _JOINT_PARAMS.items():
        if _re.fullmatch(pattern, joint_name):
            return params
    raise ValueError(f"No matching params for joint: {joint_name}")


def _match_default_pos(joint_name: str) -> float:
    """Return default position for a joint name."""
    for pattern, pos in _DEFAULT_JOINT_POS.items():
        if _re.fullmatch(pattern, joint_name):
            return pos
    return 0.0


def build_joint_arrays(joint_names: list[str]):
    """Build per-joint arrays: default_pos, kp, kd, action_scale."""
    n = len(joint_names)
    default_pos = np.zeros(n)
    kp = np.zeros(n)
    kd = np.zeros(n)
    action_scale = np.zeros(n)

    for i, name in enumerate(joint_names):
        effort, k_p, k_d = _match_joint_params(name)
        default_pos[i] = _match_default_pos(name)
        kp[i] = k_p
        kd[i] = k_d
        # action_scale = 0.25 * effort_limit / stiffness  (from g1.py)
        action_scale[i] = 0.25 * effort / k_p

    return default_pos, kp, kd, action_scale


# ─────────────────────────────────────────────────────────────
#  Policy loading (from RSL-RL .pt checkpoint)
# ─────────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, device: str = "cpu"):
    """Load actor MLP + empirical normalizer from RSL-RL checkpoint.

    Returns:
        actor: torch.nn.Module (MLP)
        normalizer: function or None
        obs_dim: int
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_dict = ckpt["model_dict"]

    # Extract actor weights — keys like "actor.0.weight", "actor.0.bias", etc.
    actor_layers = OrderedDict()
    actor_keys = sorted([k for k in model_dict if k.startswith("actor.")])
    
    # Determine layer structure
    max_idx = max(int(k.split(".")[1]) for k in actor_keys)
    
    layers = []
    for i in range(max_idx + 1):
        w_key = f"actor.{i}.weight"
        b_key = f"actor.{i}.bias"
        if w_key in model_dict:
            w = model_dict[w_key]
            b = model_dict[b_key]
            layers.append(("linear", i, w, b))

    # Build MLP
    modules = []
    for idx, (_, _, w, b) in enumerate(layers):
        in_features = w.shape[1]
        out_features = w.shape[0]
        linear = torch.nn.Linear(in_features, out_features)
        linear.weight.data = w
        linear.bias.data = b
        modules.append(linear)
        # Add activation after every linear except the last
        if idx < len(layers) - 1:
            if ACTIVATION == "elu":
                modules.append(torch.nn.ELU())
            elif ACTIVATION == "relu":
                modules.append(torch.nn.ReLU())
            elif ACTIVATION == "tanh":
                modules.append(torch.nn.Tanh())
            else:
                modules.append(torch.nn.ELU())

    actor = torch.nn.Sequential(*modules).to(device)
    actor.eval()

    obs_dim = layers[0][2].shape[1]  # input dimension of first layer
    action_dim = layers[-1][2].shape[0]  # output dimension of last layer

    print(f"[policy] Actor MLP: {obs_dim} → {ACTOR_HIDDEN_DIMS} → {action_dim}")
    print(f"[policy] {len(layers)} linear layers")

    # Load empirical normalization
    normalizer = None
    if "obs_normalizer" in ckpt and ckpt["obs_normalizer"] is not None:
        norm_state = ckpt["obs_normalizer"]
        if isinstance(norm_state, dict):
            mean = torch.tensor(norm_state.get("mean", np.zeros(obs_dim)), dtype=torch.float32)
            var = torch.tensor(norm_state.get("var", np.ones(obs_dim)), dtype=torch.float32)
            count = norm_state.get("count", 0)
            eps = 1e-6
            std = torch.sqrt(var + eps)
            normalizer = (mean.to(device), std.to(device))
            print(f"[policy] Empirical normalization: count={count}")
        else:
            print("[policy] Warning: unrecognized normalizer format, skipping")
    else:
        print("[policy] No normalizer found in checkpoint")

    return actor, normalizer, obs_dim, action_dim


def normalize_obs(obs: torch.Tensor, normalizer) -> torch.Tensor:
    if normalizer is not None:
        mean, std = normalizer
        return (obs - mean) / std
    return obs


# ─────────────────────────────────────────────────────────────
#  Math helpers
# ─────────────────────────────────────────────────────────────

def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse quaternion (w,x,y,z)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (w,x,y,z)."""
    w, x, y, z = q
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Inverse-rotate vector v by quaternion q."""
    return quat_rotate(quat_inv(q), v)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def subtract_frame_transform(
    pos1: np.ndarray, quat1: np.ndarray,
    pos2: np.ndarray, quat2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transform of frame 2 relative to frame 1.
    
    Equivalent to Isaac Lab's subtract_frame_transforms.
    Returns (pos_2_in_1, quat_2_in_1).
    """
    pos_rel = quat_rotate_inv(quat1, pos2 - pos1)
    quat_rel = quat_mul(quat_inv(quat1), quat2)
    return pos_rel, quat_rel


# ─────────────────────────────────────────────────────────────
#  Main playback loop
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play BeyondMimic policy in MuJoCo")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--motion", type=str, required=True,
                        help="Path to .npz reference motion")
    parser.add_argument("--mjcf", type=str, default=None,
                        help="Path to MJCF model (auto-detect if omitted)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (0.5 = half speed)")
    parser.add_argument("--no_normalization", action="store_true",
                        help="Disable observation normalization")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Auto-detect MJCF path
    if args.mjcf is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        args.mjcf = os.path.join(
            repo_root,
            "source", "whole_body_tracking", "whole_body_tracking",
            "assets", "unitree_description", "mjcf", "g1.xml",
        )
        print(f"[mjcf] Auto-detected: {args.mjcf}")

    # ── Load MuJoCo model ──
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    print(f"[mjoco] Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")

    # Get joint info (skip free joint at index 0)
    joint_names_dof = []
    joint_qpos_adr = []
    joint_dof_adr = []
    for i in range(1, model.njnt):  # skip free joint (i=0)
        name = model.joint(i).name
        joint_names_dof.append(name)
        joint_qpos_adr.append(model.jnt_qposadr[i])
        joint_dof_adr.append(model.jnt_dofadr[i])

    n_dof = len(joint_names_dof)
    print(f"[mjoco] DOF joints ({n_dof}): {joint_names_dof}")

    # Build per-joint arrays
    default_pos, kp, kd, action_scale = build_joint_arrays(joint_names_dof)
    print(f"[mjoco] Default pos: {default_pos}")
    print(f"[mjoco] KP range: [{kp.min():.2f}, {kp.max():.2f}]")
    print(f"[mjoco] Action scale range: [{action_scale.min():.4f}, {action_scale.max():.4f}]")

    # ── Load reference motion ──
    motion_data = np.load(args.motion)
    ref_joint_pos = motion_data["joint_pos"].astype(np.float32)   # (T, n_dof)
    ref_joint_vel = motion_data["joint_vel"].astype(np.float32)   # (T, n_dof)
    ref_body_pos_w = motion_data["body_pos_w"].astype(np.float32)  # (T, n_bodies, 3)
    ref_body_quat_w = motion_data["body_quat_w"].astype(np.float32)  # (T, n_bodies, 4)
    ref_body_lin_vel = motion_data["body_lin_vel_w"].astype(np.float32)
    ref_body_ang_vel = motion_data["body_ang_vel_w"].astype(np.float32)
    motion_fps = float(motion_data["fps"])
    n_motion_steps = ref_joint_pos.shape[0]
    print(f"[motion] Loaded: {n_motion_steps} frames, fps={motion_fps}")
    print(f"[motion] joint_pos shape: {ref_joint_pos.shape}, body_pos_w shape: {ref_body_pos_w.shape}")

    # Map body names to .npz body indices
    # The .npz stores ALL 30 bodies in PhysX BFS order (NOT the cfg body_names order).
    # This BFS ordering was verified by cross-checking left/right y-coordinates.
    NPZ_BODY_ORDER = [
        "pelvis",                    # 0
        "left_hip_pitch_link",       # 1
        "right_hip_pitch_link",      # 2
        "waist_yaw_link",            # 3
        "left_hip_roll_link",        # 4
        "right_hip_roll_link",       # 5
        "waist_roll_link",           # 6
        "left_hip_yaw_link",         # 7
        "right_hip_yaw_link",        # 8
        "torso_link",                # 9  ← ANCHOR
        "left_knee_link",            # 10
        "right_knee_link",           # 11
        "left_shoulder_pitch_link",  # 12
        "right_shoulder_pitch_link", # 13
        "left_ankle_pitch_link",     # 14
        "right_ankle_pitch_link",    # 15
        "left_shoulder_roll_link",   # 16
        "right_shoulder_roll_link",  # 17
        "left_ankle_roll_link",      # 18
        "right_ankle_roll_link",     # 19
        "left_shoulder_yaw_link",    # 20
        "right_shoulder_yaw_link",   # 21
        "left_elbow_link",           # 22
        "right_elbow_link",          # 23
        "left_wrist_roll_link",      # 24
        "right_wrist_roll_link",     # 25
        "left_wrist_pitch_link",     # 26
        "right_wrist_pitch_link",    # 27
        "left_wrist_yaw_link",       # 28
        "right_wrist_yaw_link",      # 29
    ]
    anchor_body_name = "torso_link"
    anchor_npz_idx = NPZ_BODY_ORDER.index(anchor_body_name)  # = 9

    # Get MuJoCo body ID for the anchor
    anchor_body_mj_id = model.body(anchor_body_name).id
    print(f"[motion] Anchor body '{anchor_body_name}': npz_idx={anchor_npz_idx}, mj_id={anchor_body_mj_id}")

    # ── Load policy ──
    device = "cpu"
    actor, normalizer, obs_dim, action_dim = load_policy(args.checkpoint, device)
    print(f"[policy] obs_dim={obs_dim}, action_dim={action_dim}")
    
    if action_dim != n_dof:
        print(f"[WARNING] action_dim={action_dim} != n_dof={n_dof}! Joint order may mismatch.")

    if args.no_normalization:
        normalizer = None
        print("[policy] Normalization DISABLED by user flag")

    # ── Initialize robot state ──
    # Reset to default position
    mujoco.mj_resetData(model, data)
    
    # Set free joint position
    data.qpos[0:3] = [0.0, 0.0, INIT_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion (w,x,y,z)
    data.qvel[0:6] = 0.0

    # Set joint positions to default
    for i, adr in enumerate(joint_qpos_adr):
        data.qpos[adr] = default_pos[i]
    for i, adr in enumerate(joint_dof_adr):
        data.qvel[adr] = 0.0

    mujoco.mj_forward(model, data)
    print(f"[init] Robot at height {INIT_HEIGHT}m with default joint positions")

    # ── Main playback loop ──
    motion_step = 0
    last_action = np.zeros(action_dim, dtype=np.float32)
    
    # Observation construction:
    # 1. command: ref_joint_pos (n_dof) + ref_joint_vel (n_dof) = 2*n_dof
    # 2. motion_anchor_pos_b: 3
    # 3. motion_anchor_ori_b: 6 (rotmat first 2 cols)
    # 4. base_lin_vel: 3
    # 5. base_ang_vel: 3
    # 6. joint_pos_rel: n_dof
    # 7. joint_vel: n_dof
    # 8. last_action: n_dof
    # Total: 2*n_dof + 3 + 6 + 3 + 3 + n_dof + n_dof + n_dof = 5*n_dof + 15
    
    expected_obs_dim = 5 * n_dof + 15
    print(f"[obs] Expected obs dim: {expected_obs_dim} (policy expects {obs_dim})")
    if expected_obs_dim != obs_dim:
        print(f"[WARNING] Obs dimension mismatch! Expected {expected_obs_dim}, policy expects {obs_dim}")
        print("[WARNING] The script may not work correctly. Check joint count and observation terms.")

    print("\n[INFO] Starting MuJoCo playback. Close the viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        while viewer.is_running():
            # Clamp motion step
            t = min(motion_step, n_motion_steps - 1)

            # ── Get robot state from MuJoCo ──
            # Anchor body (torso_link) world pose
            robot_anchor_pos_w = data.xpos[anchor_body_mj_id].copy()
            robot_anchor_quat_w = data.xquat[anchor_body_mj_id].copy()
            # MuJoCo qvel for free joint: [0:3] = lin vel (body frame), [3:6] = ang vel (body frame)
            base_lin_vel = data.qvel[0:3].copy()
            base_ang_vel = data.qvel[3:6].copy()
            # Joint positions and velocities
            joint_pos = np.array([data.qpos[adr] for adr in joint_qpos_adr])
            joint_vel = np.array([data.qvel[adr] for adr in joint_dof_adr])

            # ── Get reference motion at current step ──
            ref_jpos = ref_joint_pos[t]      # (n_dof,)
            ref_jvel = ref_joint_vel[t]      # (n_dof,)
            ref_anchor_pos_w = ref_body_pos_w[t, anchor_npz_idx]  # (3,)
            ref_anchor_quat_w = ref_body_quat_w[t, anchor_npz_idx]  # (4,)

            # ── Build observation vector ──
            # 1. command: ref_joint_pos + ref_joint_vel
            obs_command = np.concatenate([ref_jpos, ref_jvel])

            # 2. motion_anchor_pos_b: ref anchor in robot anchor body frame
            anchor_pos_b, anchor_quat_b = subtract_frame_transform(
                robot_anchor_pos_w, robot_anchor_quat_w,
                ref_anchor_pos_w, ref_anchor_quat_w,
            )
            obs_anchor_pos_b = anchor_pos_b  # (3,)

            # 3. motion_anchor_ori_b: first 2 columns of rotation matrix
            rotmat_b = quat_to_rotmat(anchor_quat_b)
            obs_anchor_ori_b = rotmat_b[:, :2].flatten()  # (6,)

            # 4-5. base velocities (already in body frame from MuJoCo free joint)
            obs_base_lin_vel = base_lin_vel  # (3,)
            obs_base_ang_vel = base_ang_vel  # (3,)

            # 6. joint_pos_rel: deviation from default
            obs_joint_pos_rel = joint_pos - default_pos  # (n_dof,)

            # 7. joint_vel
            obs_joint_vel = joint_vel  # (n_dof,)

            # 8. last_action
            obs_last_action = last_action  # (n_dof,)

            # Concatenate all terms
            obs_np = np.concatenate([
                obs_command,           # 2*n_dof
                obs_anchor_pos_b,      # 3
                obs_anchor_ori_b,      # 6
                obs_base_lin_vel,      # 3
                obs_base_ang_vel,      # 3
                obs_joint_pos_rel,     # n_dof
                obs_joint_vel,         # n_dof
                obs_last_action,       # n_dof
            ]).astype(np.float32)

            # ── Normalize observation ──
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            obs_normed = normalize_obs(obs_tensor, normalizer)

            # ── Policy inference ──
            with torch.no_grad():
                action_mean = actor(obs_normed).squeeze(0).cpu().numpy()

            # Use mean action (deterministic)
            action = action_mean.astype(np.float32)
            last_action = action.copy()

            # ── Compute target joint positions ──
            target_pos = default_pos + action * action_scale

            # ── PD control + simulation step (DECIMATION sub-steps) ──
            for _ in range(DECIMATION):
                # PD torque: tau = kp * (target - current) - kd * vel
                tau = kp * (target_pos - joint_pos) - kd * joint_vel
                # Clamp to effort limits
                for i, adr in enumerate(joint_dof_adr):
                    effort_limit, _, _ = _match_joint_params(joint_names_dof[i])
                    tau[i] = np.clip(tau[i], -effort_limit, effort_limit)
                
                # Set control torques
                data.ctrl[:] = tau
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Update joint readings
                joint_pos = np.array([data.qpos[adr] for adr in joint_qpos_adr])
                joint_vel = np.array([data.qvel[adr] for adr in joint_dof_adr])

            # ── Advance motion and timing ──
            motion_step += 1
            step_count += 1

            # Loop the motion
            if motion_step >= n_motion_steps:
                print(f"[playback] Motion ended at step {step_count}, looping...")
                motion_step = 0
                last_action = np.zeros(action_dim, dtype=np.float32)

            # ── Sync viewer ──
            viewer.sync()

            # ── Pace the playback ──
            if args.speed > 0:
                time.sleep(POLICY_DT / args.speed)

    print("[INFO] Playback ended.")


if __name__ == "__main__":
    main()
