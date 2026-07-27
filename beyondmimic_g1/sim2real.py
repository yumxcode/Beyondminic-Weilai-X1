from __future__ import annotations

import argparse
import time

import numpy as np

from .config import RuntimeConfig, load_runtime_config
from .math_utils import matrix_to_quat, quat_to_matrix, tilt_angle
from .motion import MotionData
from .policy import OnnxPolicy
from .runtime import ObservationBuilder, load_urdf_joint_limits, permutation


class MotionComplete(RuntimeError):
    pass


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def pelvis_to_torso_quat(
    pelvis_quat: np.ndarray,
    waist_yaw: float,
    waist_roll: float,
    waist_pitch: float,
) -> np.ndarray:
    torso = (
        quat_to_matrix(pelvis_quat)
        @ _axis_rotation("z", waist_yaw)
        @ _axis_rotation("x", waist_roll)
        @ _axis_rotation("y", waist_pitch)
    )
    return matrix_to_quat(torso)


class G1Controller:
    def __init__(self, config: RuntimeConfig, network_interface: str):
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is required for real deployment; follow the Unitree SDK2 Python installation guide"
            ) from exc

        from deploy_real.common.command_helper import MotorMode, init_cmd_hg
        from deploy_real.common.remote_controller import RemoteController

        self.config = config
        self.policy = OnnxPolicy(config.policy_path, config.raw)
        self.motion = MotionData(config.motion_path, self.policy.spec.num_actions)
        if self.motion.fps != round(1.0 / config.control_dt):
            raise ValueError("motion FPS must match the configured real-time control rate")
        self.runtime_names = config.runtime_joint_names
        self.motor_indices = config.motor_indices
        self.policy_to_runtime = permutation(list(self.policy.spec.joint_names), self.runtime_names)
        self.default_runtime = self.policy.spec.default_joint_pos[self.policy_to_runtime]
        self.kp_runtime = self.policy.spec.stiffness[self.policy_to_runtime]
        self.kd_runtime = self.policy.spec.damping[self.policy_to_runtime]
        self.lower, self.upper = load_urdf_joint_limits(config)
        self.target = np.clip(self.default_runtime.copy(), self.lower, self.upper)
        self.builder = ObservationBuilder(
            self.policy.spec,
            tuple(self.runtime_names),
            int(config.raw["robot"]["motion_anchor_body_index"]),
        )
        self.remote = RemoteController()
        self._crc = CRC()
        self._last_state_time = 0.0
        self._mode_machine = 0
        self._low_state_type = LowState_
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.publisher = ChannelPublisher(config.raw["robot"]["lowcmd_topic"], LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber(config.raw["robot"]["lowstate_topic"], LowState_)
        self.subscriber.Init(self._state_callback, 10)
        self._wait_for_state()
        init_cmd_hg(self.low_cmd, self._mode_machine, MotorMode.PR)

    def _state_callback(self, message) -> None:
        self.low_state = message
        self._mode_machine = message.mode_machine
        self.remote.set(message.wireless_remote)
        self._last_state_time = time.monotonic()

    def _wait_for_state(self) -> None:
        timeout = float(self.config.raw["safety"]["low_state_timeout_s"]) * 10
        deadline = time.monotonic() + timeout
        while self._last_state_time == 0.0:
            if time.monotonic() >= deadline:
                raise TimeoutError("no rt/lowstate message received; check network interface and robot debug mode")
            time.sleep(0.01)

    def _check_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        age = time.monotonic() - self._last_state_time
        timeout = float(self.config.raw["safety"]["low_state_timeout_s"])
        if age > timeout:
            raise TimeoutError(f"low-state stream timed out ({age:.3f}s)")
        q = np.asarray([self.low_state.motor_state[index].q for index in self.motor_indices], dtype=np.float32)
        dq = np.asarray([self.low_state.motor_state[index].dq for index in self.motor_indices], dtype=np.float32)
        pelvis_quat = np.asarray(self.low_state.imu_state.quaternion, dtype=np.float64)
        gyro_b = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32).reshape(3)
        if not all(np.all(np.isfinite(value)) for value in (q, dq, pelvis_quat, gyro_b)):
            raise RuntimeError("robot state contains NaN or infinity")
        if np.max(np.abs(dq)) > float(self.config.raw["safety"]["max_joint_velocity_rad_s"]):
            raise RuntimeError("joint velocity safety threshold exceeded")
        if tilt_angle(pelvis_quat) > float(self.config.raw["safety"]["max_tilt_rad"]):
            raise RuntimeError("robot tilt safety threshold exceeded")
        return q, dq, pelvis_quat, gyro_b

    def _write_position_command(self, target: np.ndarray) -> None:
        self.low_cmd.mode_machine = self._mode_machine
        for runtime_index, motor_index in enumerate(self.motor_indices):
            command = self.low_cmd.motor_cmd[motor_index]
            command.q = float(target[runtime_index])
            command.qd = 0.0
            command.kp = float(self.kp_runtime[runtime_index])
            command.kd = float(self.kd_runtime[runtime_index])
            command.tau = 0.0
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def send_zero(self) -> None:
        from deploy_real.common.command_helper import create_zero_cmd

        create_zero_cmd(self.low_cmd)
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def send_damping(self, repeats: int = 10) -> None:
        from deploy_real.common.command_helper import create_damping_cmd

        create_damping_cmd(self.low_cmd)
        for _ in range(repeats):
            self.low_cmd.crc = self._crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)
            time.sleep(self.config.control_dt)

    def await_start(self) -> None:
        from deploy_real.common.remote_controller import KeyMap

        print("Zero-torque state. Press START to move to the default pose; SELECT aborts.")
        while not self.remote.button[KeyMap.start]:
            if self.remote.button[KeyMap.select]:
                raise KeyboardInterrupt
            self.send_zero()
            time.sleep(self.config.control_dt)

    def move_to_default(self) -> None:
        q, _, _, _ = self._check_state()
        steps = max(1, round(float(self.config.raw["safety"]["start_transition_s"]) / self.config.control_dt))
        print(f"Moving to default pose over {steps * self.config.control_dt:.1f}s.")
        for step in range(steps):
            alpha = (step + 1) / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self._write_position_command(q * (1.0 - smooth) + self.default_runtime * smooth)
            time.sleep(self.config.control_dt)

    def await_policy_enable(self) -> None:
        from deploy_real.common.remote_controller import KeyMap

        print("Holding default pose. Press A to start the motion; SELECT aborts.")
        while not self.remote.button[KeyMap.A]:
            if self.remote.button[KeyMap.select]:
                raise KeyboardInterrupt
            self._check_state()
            self._write_position_command(self.default_runtime)
            time.sleep(self.config.control_dt)

    def run_policy(self) -> None:
        from deploy_real.common.remote_controller import KeyMap

        first = self.motion.frame(0)
        q, dq, pelvis_quat, gyro_b = self._check_state()
        torso_quat = self._torso_quat(pelvis_quat, q)
        self.builder.reset(torso_quat, first.body_quat_w[self.builder.anchor_body_index])
        action_clip = float(self.config.raw["safety"]["action_clip"])
        max_overruns = int(self.config.raw["safety"]["max_control_overrun_count"])
        overrun_count = 0
        print("Policy enabled. Press SELECT for damping mode.")
        for time_step in range(self.motion.num_frames):
            tick = time.perf_counter()
            if self.remote.button[KeyMap.select]:
                raise KeyboardInterrupt
            q, dq, pelvis_quat, gyro_b = self._check_state()
            torso_quat = self._torso_quat(pelvis_quat, q)
            observation = self.builder.build(
                self.motion.frame(time_step),
                torso_quat,
                gyro_b,
                q,
                dq,
            )
            action = self.policy.infer(observation, time_step)
            self.target = np.clip(self.builder.target_runtime(action, action_clip), self.lower, self.upper)
            self._write_position_command(self.target)
            remaining = self.config.control_dt - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
                overrun_count = 0
            else:
                overrun_count += 1
                if overrun_count >= max_overruns:
                    raise RuntimeError("real-time control loop repeatedly missed its deadline")
        raise MotionComplete("reference motion completed")

    def _torso_quat(self, pelvis_quat: np.ndarray, q_runtime: np.ndarray) -> np.ndarray:
        by_name = dict(zip(self.runtime_names, q_runtime))
        return pelvis_to_torso_quat(
            pelvis_quat,
            by_name["waist_yaw_joint"],
            by_name["waist_roll_joint"],
            by_name["waist_pitch_joint"],
        )


def run(args: argparse.Namespace) -> None:
    if not args.enable_real:
        raise SystemExit(
            "Refusing to publish motor commands without --enable-real. "
            "First validate the same policy with sim2sim and use a safety suspension."
        )
    config = load_runtime_config(args.config).with_overrides(policy_path=args.policy, motion_path=args.motion)
    config.validate()
    if int(config.raw["policy"]["num_observations"]) != 154:
        raise ValueError("sim2real currently supports only the 154-D no-state-estimation policy")

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ImportError as exc:
        raise RuntimeError("unitree_sdk2py is not installed") from exc
    ChannelFactoryInitialize(0, args.network_interface)
    controller: G1Controller | None = None
    try:
        controller = G1Controller(config, args.network_interface)
        controller.await_start()
        controller.move_to_default()
        controller.await_policy_enable()
        controller.run_policy()
    except MotionComplete as exc:
        print(str(exc))
    except KeyboardInterrupt:
        print("Operator stop requested.")
    finally:
        if controller is not None:
            print("Entering damping mode.")
            controller.send_damping()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy a BeyondMimic ONNX policy to a 29-DoF Unitree G1.")
    parser.add_argument("network_interface", help="Network interface connected to G1, for example enp3s0")
    parser.add_argument("--config", default="configs/g1_beyondmimic.yaml")
    parser.add_argument("--policy", help="Override policy.path from the configuration")
    parser.add_argument("--motion", help="Override policy.motion_path from the configuration")
    parser.add_argument(
        "--enable-real",
        action="store_true",
        help="Required acknowledgement that this command will publish low-level motor commands",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
