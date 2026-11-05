#!/usr/bin/env python3
"""Sliding-window travel scan over LAFAN dance BVHs.

For each candidate window (default 750 frames = 25 s, stride 150), compute
in-window root xy bbox, displacement and path length. Rank windows by
"travel" = min(bbox_x, bbox_y-ish) metric: we want the root to cross space
inside the window (either axis >= 2 m for the human).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tools_scan_lafan_travel import parse_bvh_frames


def main() -> None:
    win = int(sys.argv[1]) if len(sys.argv) > 1 else 750
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    patterns = sys.argv[3:] or ["dance*.bvh"]
    root = Path(__file__).resolve().parent / "downloads" / "lafan1"
    files = sorted({f for pat in patterns for f in root.glob(pat)})

    results = []
    for f in files:
        pos_cm, ft = parse_bvh_frames(f)
        xy = pos_cm[:, :2] / 100.0
        n = len(xy)
        for start in range(0, n - win + 1, stride):
            seg = xy[start : start + win]
            bbox = seg.max(axis=0) - seg.min(axis=0)
            displ = float(np.linalg.norm(seg[-1] - seg[0]))
            d = np.diff(seg, axis=0)
            path = float(np.linalg.norm(d, axis=1).sum())
            # net travel: distance between window endpoints AND bbox support
            travel = float(np.linalg.norm(seg[-1] - seg[0]))
            results.append((travel, f.name, start, bbox[0], bbox[1], displ, path))

    results.sort(reverse=True)
    print(f"top windows (win={win}f/{win/30:.1f}s, stride={stride})")
    print(f"{'file':26s} {'start':>6s} {'end':>6s} {'trav_m':>7s} {'bbox_x':>7s} {'bbox_y':>7s} {'path_m':>7s}")
    for travel, name, start, bx, by, displ, path in results[:25]:
        print(f"{name:26s} {start:6d} {start+win:6d} {travel:7.2f} {bx:7.2f} {by:7.2f} {path:7.2f}")


if __name__ == "__main__":
    main()
