from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .config import RuntimeConfig
from .math_utils import initial_motion_to_world_yaw, relative_orientation_6d
from .motion import MotionFrame
from .policy import PolicySpec


def permutation(source_names: list[str] | tuple[str, ...], target_names: list[str] | tuple[str, ...]) -> np.ndarray:
    if set(source_names) != set(target_names):
        missing = sorted(set(target_names) - set(source_names))
        extra = sorted(set(source_names) - set(target_names))
        raise ValueError(f"joint name mismatch; missing={missing}, extra={extra}")
    return np.asarray([source_names.index(name) for name in target_names], dtype=np.int64)


def load_urdf_joint_limits(config: RuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
    joints = {
        joint.attrib["name"]: joint.find("limit")
        for joint in ET.parse(config.urdf_path).getroot().findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous"}
    }
    lower: list[float] = []
    upper: list[float] = []
    for name in config.runtime_joint_names:
        limit = joints.get(name)
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"URDF has no position limit for {name}")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
    margin = float(config.raw["safety"]["joint_limit_margin_rad"])
    lower_array = np.asarray(lower, dtype=np.float32) + margin
    upper_array = np.asarray(upper, dtype=np.float32) - margin
    if np.any(lower_array >= upper_array):
        raise ValueError("joint limit margin is too large")
    return lower_array, upper_array


def load_urdf_effort_limits(config: RuntimeConfig) -> np.ndarray:
    joints = {
        joint.attrib["name"]: joint.find("limit")
        for joint in ET.parse(config.urdf_path).getroot().findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous"}
    }
    effort: list[float] = []
    for name in config.runtime_joint_names:
        limit = joints.get(name)
        if limit is None or "effort" not in limit.attrib:
            raise ValueError(f"URDF has no effort limit for {name}")
        effort.append(float(limit.attrib["effort"]))
    return np.asarray(effort, dtype=np.float32)


@dataclass
class ObservationBuilder:
    spec: PolicySpec
    runtime_joint_names: tuple[str, ...]
    anchor_body_index: int

    def __post_init__(self):
        self._runtime_to_policy = permutation(list(self.runtime_joint_names), list(self.spec.joint_names))
        self._motion_to_world_yaw: np.ndarray | None = None
        self.last_action = np.zeros(self.spec.num_actions, dtype=np.float32)

    def reset(self, robot_anchor_quat_w: np.ndarray, motion_anchor_quat_w: np.ndarray) -> None:
        self._motion_to_world_yaw = initial_motion_to_world_yaw(robot_anchor_quat_w, motion_anchor_quat_w)
        self.last_action.fill(0.0)

    def build(
        self,
        motion: MotionFrame,
        robot_anchor_quat_w: np.ndarray,
        base_ang_vel_b: np.ndarray,
        joint_pos_runtime: np.ndarray,
        joint_vel_runtime: np.ndarray,
        *,
        motion_anchor_pos_b: np.ndarray | None = None,
        base_lin_vel_b: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._motion_to_world_yaw is None:
            self.reset(robot_anchor_quat_w, motion.body_quat_w[self.anchor_body_index])
        joint_pos_policy = np.asarray(joint_pos_runtime, dtype=np.float32)[self._runtime_to_policy]
        joint_vel_policy = np.asarray(joint_vel_runtime, dtype=np.float32)[self._runtime_to_policy]
        terms = [
            np.asarray(motion.joint_pos, dtype=np.float32),
            np.asarray(motion.joint_vel, dtype=np.float32),
        ]
        if self.spec.num_observations == 160:
            if motion_anchor_pos_b is None or base_lin_vel_b is None:
                raise ValueError("160-dimensional policy requires anchor position and base linear velocity estimates")
            terms.append(np.asarray(motion_anchor_pos_b, dtype=np.float32).reshape(3))
        terms.append(
            relative_orientation_6d(
                robot_anchor_quat_w,
                motion.body_quat_w[self.anchor_body_index],
                self._motion_to_world_yaw,
            )
        )
        if self.spec.num_observations == 160:
            terms.append(np.asarray(base_lin_vel_b, dtype=np.float32).reshape(3))
        terms.extend(
            [
                np.asarray(base_ang_vel_b, dtype=np.float32).reshape(3),
                joint_pos_policy - self.spec.default_joint_pos,
                joint_vel_policy,
                self.last_action,
            ]
        )
        observation = np.concatenate(terms).astype(np.float32, copy=False)
        if observation.shape != (self.spec.num_observations,) or not np.all(np.isfinite(observation)):
            raise RuntimeError(f"invalid observation shape/data: {observation.shape}")
        return observation

    def target_runtime(self, action_policy: np.ndarray, action_clip: float) -> np.ndarray:
        action = np.clip(np.asarray(action_policy, dtype=np.float32), -action_clip, action_clip)
        self.last_action = action.copy()
        target_policy = self.spec.default_joint_pos + action * self.spec.action_scale
        policy_to_runtime = permutation(list(self.spec.joint_names), list(self.runtime_joint_names))
        return target_policy[policy_to_runtime]
