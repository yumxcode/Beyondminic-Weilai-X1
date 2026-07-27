from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    raw = str(value).replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def control_dt(self) -> float:
        return float(self.raw["runtime"]["control_dt"])

    @property
    def simulation_dt(self) -> float:
        return float(self.raw["runtime"]["simulation_dt"])

    @property
    def control_decimation(self) -> int:
        ratio = self.control_dt / self.simulation_dt
        rounded = round(ratio)
        if abs(ratio - rounded) > 1.0e-9:
            raise ValueError("control_dt must be an integer multiple of simulation_dt")
        return int(rounded)

    @property
    def policy_path(self) -> Path:
        return _resolve_path(self.raw["policy"]["path"], self.path)

    @property
    def motion_path(self) -> Path:
        return _resolve_path(self.raw["policy"]["motion_path"], self.path)

    @property
    def mjcf_path(self) -> Path:
        return _resolve_path(self.raw["robot"]["mjcf_path"], self.path)

    @property
    def urdf_path(self) -> Path:
        return _resolve_path(self.raw["robot"]["urdf_path"], self.path)

    @property
    def runtime_joint_names(self) -> list[str]:
        return list(self.raw["robot"]["joint_names"])

    @property
    def motor_indices(self) -> list[int]:
        return [int(x) for x in self.raw["robot"]["motor_indices"]]

    def with_overrides(
        self,
        *,
        policy_path: str | Path | None = None,
        motion_path: str | Path | None = None,
    ) -> "RuntimeConfig":
        import copy

        raw = copy.deepcopy(self.raw)
        if policy_path is not None:
            raw["policy"]["path"] = str(Path(policy_path).expanduser().resolve())
        if motion_path is not None:
            raw["policy"]["motion_path"] = str(Path(motion_path).expanduser().resolve())
        return RuntimeConfig(self.path, raw)

    def validate(self, require_files: bool = True) -> None:
        expected_actions = int(self.raw["policy"]["num_actions"])
        expected_obs = int(self.raw["policy"]["num_observations"])
        if expected_actions != 29:
            raise ValueError(f"this G1 runtime requires 29 actions, got {expected_actions}")
        if expected_obs not in (154, 160):
            raise ValueError(f"supported observation sizes are 154 and 160, got {expected_obs}")
        if len(self.runtime_joint_names) != expected_actions:
            raise ValueError("robot.joint_names length does not match policy.num_actions")
        if len(set(self.runtime_joint_names)) != expected_actions:
            raise ValueError("robot.joint_names contains duplicates")
        if len(self.motor_indices) != expected_actions or len(set(self.motor_indices)) != expected_actions:
            raise ValueError("robot.motor_indices must contain 29 unique entries")
        for name in ("stiffness", "damping", "default_joint_pos", "action_scale"):
            values = self.raw["control"][name]
            if len(values) != expected_actions:
                raise ValueError(f"control.{name} must contain {expected_actions} values")
        if self.control_dt <= 0 or self.simulation_dt <= 0:
            raise ValueError("control time steps must be positive")
        _ = self.control_decimation
        if require_files:
            for label, path in (
                ("policy", self.policy_path),
                ("motion", self.motion_path),
                ("MJCF", self.mjcf_path),
                ("URDF", self.urdf_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"{label} file not found: {path}")


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: python -m pip install PyYAML") from exc

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid runtime config: {config_path}")
    config = RuntimeConfig(config_path, raw)
    config.validate(require_files=False)
    return config
