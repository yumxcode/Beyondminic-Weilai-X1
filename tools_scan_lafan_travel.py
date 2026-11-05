#!/usr/bin/env python3
"""Scan LAFAN1 BVH files for root-travel statistics (xy displacement).

Parses the MOTION section directly (no FK): Hips channels are
Xpos Ypos Zpos Zrot Yrot Xrot, so per-frame root translation = first 3 floats
of each motion line (cm in LAFAN1).

Reports per file: duration, xy bounding box, start->end distance, path
length, mean speed. Threshold highlight for "traveling" candidates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def parse_bvh_frames(path: Path) -> tuple[np.ndarray, float]:
    lines = path.read_text().splitlines()
    n_frames = None
    frame_time = None
    motion_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Frames:"):
            n_frames = int(s.split(":")[1])
        elif s.startswith("Frame Time:"):
            frame_time = float(s.split(":")[1])
            motion_start = i + 1
            break
    if n_frames is None or frame_time is None:
        raise ValueError(f"{path}: no MOTION header")
    rows = []
    for line in lines[motion_start : motion_start + n_frames]:
        vals = [float(v) for v in line.split()]
        if len(vals) >= 3:
            rows.append(vals[:3])
    return np.array(rows), frame_time


def main() -> None:
    patterns = sys.argv[1:] or ["dance*.bvh", "multiple*.bvh"]
    root = Path(__file__).resolve().parent / "downloads" / "lafan1"
    files = sorted({f for pat in patterns for f in root.glob(pat)})
    print(f"{'file':32s} {'frames':>7s} {'sec':>6s} {'bbox_x':>7s} {'bbox_y':>7s} "
          f"{'displ_m':>8s} {'path_m':>7s} {'speed':>6s}")
    for f in files:
        pos_cm, ft = parse_bvh_frames(f)
        n = len(pos_cm)
        dur = n * ft
        xy = pos_cm[:, :2] / 100.0  # cm -> m
        bbox = xy.max(axis=0) - xy.min(axis=0)
        displ = float(np.linalg.norm(xy[-1] - xy[0]))
        d = np.diff(xy, axis=0)
        path = float(np.linalg.norm(d, axis=1).sum())
        print(f"{f.name:32s} {n:7d} {dur:6.1f} {bbox[0]:7.2f} {bbox[1]:7.2f} "
              f"{displ:8.2f} {path:7.2f} {path / dur:6.2f}")


if __name__ == "__main__":
    main()
