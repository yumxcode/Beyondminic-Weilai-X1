from __future__ import annotations

import argparse
from contextlib import nullcontext
import time

import numpy as np

from .config import load_runtime_config
from .math_utils import initial_motion_to_world_yaw, quat_to_matrix, rotate_inverse
from .motion import MotionData
from .policy import OnnxPolicy
from .runtime import ObservationBuilder, load_urdf_effort_limits, load_urdf_joint_limits, permutation


def _joint_addresses(model, mujoco, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses = []
    qvel_addresses = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MJCF joint not found: {name}")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        qvel_addresses.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_addresses), np.asarray(qvel_addresses)


def _actuator_ids(model, mujoco, names: list[str]) -> np.ndarray:
    result = []
    for name in names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"MJCF actuator not found: {name}")
        result.append(actuator_id)
    return np.asarray(result)


def run(args: argparse.Namespace) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required: python -m pip install '.[sim]'") from exc

    config = load_runtime_config(args.config).with_overrides(policy_path=args.policy, motion_path=args.motion)
    config.validate()
    policy = OnnxPolicy(config.policy_path, config.raw)
    motion = MotionData(config.motion_path, policy.spec.num_actions)
    if motion.fps != round(1.0 / config.control_dt):
        raise ValueError(f"motion fps {motion.fps} does not match control rate {1.0 / config.control_dt:g} Hz")

    runtime_names = config.runtime_joint_names
    policy_to_runtime = permutation(list(policy.spec.joint_names), runtime_names)
    kp_runtime = policy.spec.stiffness[policy_to_runtime]
    kd_runtime = policy.spec.damping[policy_to_runtime]
    lower, upper = load_urdf_joint_limits(config)
    effort = load_urdf_effort_limits(config)

    model = mujoco.MjModel.from_xml_path(str(config.mjcf_path))
    data = mujoco.MjData(model)
    model.opt.timestep = config.simulation_dt
    qpos_adr, qvel_adr = _joint_addresses(model, mujoco, runtime_names)
    actuator_ids = _actuator_ids(model, mujoco, runtime_names)
    anchor_name = config.raw["robot"]["sim_anchor_body_name"]
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_name)
    if anchor_id < 0:
        raise ValueError(f"MJCF body not found: {anchor_name}")

    first_frame = motion.frame(0)
    data.qpos[:3] = first_frame.body_pos_w[0]
    data.qpos[2] += float(args.spawn_height_offset)
    data.qpos[3:7] = first_frame.body_quat_w[0]
    data.qpos[qpos_adr] = policy.spec.default_joint_pos[policy_to_runtime]
    mujoco.mj_forward(model, data)

    builder = ObservationBuilder(
        policy.spec,
        tuple(runtime_names),
        int(config.raw["robot"]["motion_anchor_body_index"]),
    )
    builder.reset(data.xquat[anchor_id], first_frame.body_quat_w[builder.anchor_body_index])
    motion_to_world = initial_motion_to_world_yaw(
        data.xquat[anchor_id],
        first_frame.body_quat_w[builder.anchor_body_index],
    )
    initial_robot_anchor_pos = data.xpos[anchor_id].copy()
    initial_motion_anchor_pos = first_frame.body_pos_w[builder.anchor_body_index].copy()
    target = data.qpos[qpos_adr].copy()
    action_clip = float(config.raw["safety"]["action_clip"])
    max_steps = (
        args.steps
        if args.steps is not None
        else int(config.raw["runtime"]["simulation_duration"] / config.simulation_dt)
    )
    if max_steps <= 0:
        raise ValueError("--steps must be positive")
    viewer_context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)

    control_index = 0
    overrun_count = 0
    with viewer_context as viewer:
        for simulation_step in range(max_steps):
            start = time.perf_counter()
            if simulation_step % config.control_decimation == 0:
                if control_index >= motion.num_frames:
                    behavior = config.raw["safety"]["motion_end_behavior"]
                    if behavior == "loop":
                        control_index = 0
                        builder.reset(data.xquat[anchor_id], first_frame.body_quat_w[builder.anchor_body_index])
                    elif behavior in {"stop", "damping"}:
                        break
                    else:
                        control_index = motion.num_frames - 1
                frame = motion.frame(control_index)
                pelvis_quat = data.qpos[3:7].copy()
                base_ang_vel_b = rotate_inverse(pelvis_quat, data.qvel[3:6])
                anchor_pos_b = None
                base_lin_vel_b = None
                if policy.spec.num_observations == 160:
                    aligned_delta = quat_to_matrix(motion_to_world) @ (
                        frame.body_pos_w[builder.anchor_body_index] - initial_motion_anchor_pos
                    )
                    desired_anchor_w = initial_robot_anchor_pos + aligned_delta
                    anchor_pos_b = quat_to_matrix(data.xquat[anchor_id]).T @ (
                        desired_anchor_w - data.xpos[anchor_id]
                    )
                    base_lin_vel_b = rotate_inverse(pelvis_quat, data.qvel[:3])
                observation = builder.build(
                    frame,
                    data.xquat[anchor_id],
                    base_ang_vel_b,
                    data.qpos[qpos_adr],
                    data.qvel[qvel_adr],
                    motion_anchor_pos_b=anchor_pos_b,
                    base_lin_vel_b=base_lin_vel_b,
                )
                action = policy.infer(observation, control_index)
                target = np.clip(builder.target_runtime(action, action_clip), lower, upper)
                control_index += 1

            torque = kp_runtime * (target - data.qpos[qpos_adr]) - kd_runtime * data.qvel[qvel_adr]
            torque = np.clip(torque, -effort, effort)
            data.ctrl[actuator_ids] = torque
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                remaining = config.simulation_dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    overrun_count += 1

    print(
        f"sim2sim complete: simulation_steps={simulation_step + 1}, "
        f"policy_steps={control_index}, overruns={overrun_count}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BeyondMimic G1 ONNX policy in MuJoCo.")
    parser.add_argument("--config", default="configs/g1_beyondmimic.yaml")
    parser.add_argument("--policy", help="Override policy.path from the configuration")
    parser.add_argument("--motion", help="Override policy.motion_path from the configuration")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer")
    parser.add_argument("--steps", type=int, help="Maximum number of MuJoCo simulation steps")
    parser.add_argument("--spawn-height-offset", type=float, default=0.02)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
