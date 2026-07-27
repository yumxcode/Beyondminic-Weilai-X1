from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _csv_strings(value: str | None) -> list[str]:
    return [] if not value else [item.strip() for item in value.split(",") if item.strip()]


def _csv_floats(value: str | None) -> np.ndarray | None:
    values = _csv_strings(value)
    return None if not values else np.asarray([float(item) for item in values], dtype=np.float32)


@dataclass(frozen=True)
class PolicySpec:
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    action_scale: np.ndarray
    observation_names: tuple[str, ...]
    anchor_body_name: str
    num_observations: int
    num_actions: int

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        fallback_joint_names: Sequence[str],
        fallback_control: Mapping[str, Sequence[float]],
        num_observations: int,
        num_actions: int,
    ) -> "PolicySpec":
        joint_names = tuple(_csv_strings(metadata.get("joint_names")) or fallback_joint_names)

        def values(metadata_key: str, fallback_key: str) -> np.ndarray:
            result = _csv_floats(metadata.get(metadata_key))
            if result is None:
                result = np.asarray(fallback_control[fallback_key], dtype=np.float32)
            if result.shape != (num_actions,):
                raise ValueError(f"{metadata_key} must contain {num_actions} values, got {result.shape}")
            return result

        if len(joint_names) != num_actions or len(set(joint_names)) != num_actions:
            raise ValueError(f"joint_names must contain {num_actions} unique names")
        observation_names = tuple(_csv_strings(metadata.get("observation_names")))
        return cls(
            joint_names=joint_names,
            default_joint_pos=values("default_joint_pos", "default_joint_pos"),
            stiffness=values("joint_stiffness", "stiffness"),
            damping=values("joint_damping", "damping"),
            action_scale=values("action_scale", "action_scale"),
            observation_names=observation_names,
            anchor_body_name=metadata.get("anchor_body_name", "torso_link"),
            num_observations=num_observations,
            num_actions=num_actions,
        )


class OnnxPolicy:
    def __init__(self, path: str | Path, config: Mapping):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required: python -m pip install onnxruntime") from exc

        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {self.path}")
        providers = config["policy"].get("providers", ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        inputs = {entry.name: entry for entry in self.session.get_inputs()}
        outputs = {entry.name: entry for entry in self.session.get_outputs()}
        if "obs" not in inputs or "time_step" not in inputs:
            raise ValueError("ONNX policy must expose inputs named 'obs' and 'time_step'")
        if "actions" not in outputs:
            raise ValueError("ONNX policy must expose an output named 'actions'")
        self.input_names = tuple(inputs)
        self.output_names = tuple(outputs)
        num_observations = int(config["policy"]["num_observations"])
        num_actions = int(config["policy"]["num_actions"])
        obs_shape = inputs["obs"].shape
        if obs_shape[-1] not in (None, "None", num_observations):
            raise ValueError(f"ONNX obs size {obs_shape[-1]} does not match config {num_observations}")
        action_shape = outputs["actions"].shape
        if action_shape[-1] not in (None, "None", num_actions):
            raise ValueError(f"ONNX action size {action_shape[-1]} does not match config {num_actions}")
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.spec = PolicySpec.from_metadata(
            metadata,
            fallback_joint_names=config["robot"]["policy_joint_names"],
            fallback_control=config["control"],
            num_observations=num_observations,
            num_actions=num_actions,
        )

    def infer(self, observation: np.ndarray, time_step: int) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, self.spec.num_observations)
        time_input = np.asarray([[time_step]], dtype=np.float32)
        result = self.session.run(["actions"], {"obs": observation, "time_step": time_input})[0]
        action = np.asarray(result, dtype=np.float32).reshape(-1)
        if action.shape != (self.spec.num_actions,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"policy returned invalid action with shape {action.shape}")
        return action
