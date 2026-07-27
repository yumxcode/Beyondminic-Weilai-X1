from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class MotionFrame:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


class MotionData:
    def __init__(self, path: str | Path, num_actions: int = 29):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"motion file not found: {self.path}")
        archive = np.load(self.path, allow_pickle=False)
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"motion file is missing keys: {', '.join(missing)}")
        self._data = {key: np.asarray(archive[key], dtype=np.float32) for key in REQUIRED_KEYS}
        self.fps = int(np.asarray(archive["fps"]).reshape(-1)[0]) if "fps" in archive.files else 50
        self.num_frames = int(self._data["joint_pos"].shape[0])
        if self.num_frames < 2:
            raise ValueError("motion must contain at least two frames")
        if self._data["joint_pos"].shape != (self.num_frames, num_actions):
            raise ValueError(f"joint_pos must have shape (frames, {num_actions})")
        if self._data["joint_vel"].shape != (self.num_frames, num_actions):
            raise ValueError(f"joint_vel must have shape (frames, {num_actions})")
        for key in REQUIRED_KEYS[2:]:
            if self._data[key].shape[0] != self.num_frames:
                raise ValueError(f"{key} frame count does not match joint_pos")
        if self._data["body_pos_w"].ndim != 3 or self._data["body_pos_w"].shape[-1] != 3:
            raise ValueError("body_pos_w must have shape (frames, bodies, 3)")
        if self._data["body_quat_w"].shape[:2] != self._data["body_pos_w"].shape[:2]:
            raise ValueError("body_quat_w body dimensions do not match body_pos_w")
        if self._data["body_quat_w"].shape[-1] != 4:
            raise ValueError("body_quat_w must contain wxyz quaternions")

    @property
    def num_bodies(self) -> int:
        return int(self._data["body_pos_w"].shape[1])

    def frame(self, index: int) -> MotionFrame:
        if not 0 <= index < self.num_frames:
            raise IndexError(f"motion frame {index} is outside [0, {self.num_frames - 1}]")
        return MotionFrame(**{key: value[index] for key, value in self._data.items()})
