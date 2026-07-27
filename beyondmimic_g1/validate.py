from __future__ import annotations

import argparse

from .config import load_runtime_config
from .motion import MotionData
from .policy import OnnxPolicy
from .runtime import load_urdf_joint_limits, permutation


def run(config_path: str, policy_path: str | None, motion_path: str | None) -> None:
    config = load_runtime_config(config_path).with_overrides(policy_path=policy_path, motion_path=motion_path)
    config.validate()
    policy = OnnxPolicy(config.policy_path, config.raw)
    motion = MotionData(config.motion_path, policy.spec.num_actions)
    permutation(config.runtime_joint_names, list(policy.spec.joint_names))
    lower, upper = load_urdf_joint_limits(config)
    anchor_index = int(config.raw["robot"]["motion_anchor_body_index"])
    if not 0 <= anchor_index < motion.num_bodies:
        raise ValueError(f"motion anchor body index {anchor_index} exceeds {motion.num_bodies} bodies")
    expected_rate = round(1.0 / config.control_dt)
    if motion.fps != expected_rate:
        raise ValueError(f"motion is {motion.fps} FPS but controller is {expected_rate} Hz")
    expected_terms = (
        ("command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions")
        if policy.spec.num_observations == 154
        else (
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        )
    )
    if policy.spec.observation_names and policy.spec.observation_names != expected_terms:
        raise ValueError(
            "ONNX observation layout is incompatible: "
            f"expected {expected_terms}, got {policy.spec.observation_names}"
        )
    print(f"OK policy: {config.policy_path}")
    print(f"   observations={policy.spec.num_observations}, actions={policy.spec.num_actions}")
    print(f"   anchor={policy.spec.anchor_body_name}, metadata_terms={policy.spec.observation_names or 'fallback'}")
    print(f"OK motion: {config.motion_path}")
    print(f"   frames={motion.num_frames}, fps={motion.fps}, bodies={motion.num_bodies}")
    print(f"OK robot: {config.urdf_path}")
    print(f"   joints={len(config.runtime_joint_names)}, safe_range=[{lower.min():.3f}, {upper.max():.3f}] rad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a BeyondMimic G1 deployment bundle.")
    parser.add_argument("--config", default="configs/g1_beyondmimic.yaml")
    parser.add_argument("--policy")
    parser.add_argument("--motion")
    args = parser.parse_args()
    run(args.config, args.policy, args.motion)


if __name__ == "__main__":
    main()
