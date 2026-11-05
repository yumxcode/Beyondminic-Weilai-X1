#!/usr/bin/env python3
"""Sliding-window scan for TRAVELING but GROUND-CONSISTENT dance windows.

The first travel scan picked windows dominated by rolls/jumps (root z swings
+-2 m). The verified-clean window (dance1_subject3 [0,689), BVH units) has
root z std 0.107, range 1.72..2.07. We now require:

  - z平稳: std <= 0.22, per-frame |dz| <= 0.06 (no jumps/rolls)
  - z 高度带: window z_min >= 1.15 (crouch allowed, floor work excluded)
  - travel: net displacement >= 2.5 m (human, BVH unit ~m after /100)
Ranks by travel.
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
        xyz = pos_cm / 100.0
        n = len(xyz)
        xy, z = xyz[:, :2], xyz[:, 2]
        dz = np.abs(np.diff(z))
        for start in range(0, n - win + 1, stride):
            seg_xy = xy[start : start + win]
            seg_z = z[start : start + win]
            travel = float(np.linalg.norm(seg_xy[-1] - seg_xy[0]))
            z_std = float(seg_z.std())
            z_min = float(seg_z.min())
            max_dz = float(dz[start : start + win - 1].max())
            ok = (z_std <= 0.22) and (z_min >= 1.15) and (max_dz <= 0.06) and (travel >= 2.5)
            results.append((ok, travel, f.name, start, z_std, z_min, max_dz))

    good = [r for r in results if r[0]]
    good.sort(key=lambda r: -r[1])
    print(f"windows passing z-consistency + travel>=2.5m: {len(good)} / {len(results)}")
    print(f"{'file':26s} {'start':>6s} {'end':>6s} {'trav_m':>7s} {'z_std':>6s} {'z_min':>6s} {'maxdz':>6s}")
    for ok, travel, name, start, z_std, z_min, max_dz in good[:20]:
        print(f"{name:26s} {start:6d} {start+win:6d} {travel:7.2f} {z_std:6.3f} {z_min:6.2f} {max_dz:6.3f}")
    if not good:
        print("--- near misses (travel>=2.0, relaxed dz<=0.08) ---")
        near = [r for r in results if r[1] >= 2.0 and r[4] <= 0.30 and r[5] >= 1.0 and r[6] <= 0.08]
        near.sort(key=lambda r: -r[1])
        for ok, travel, name, start, z_std, z_min, max_dz in near[:15]:
            print(f"{name:26s} {start:6d} {start+win:6d} {travel:7.2f} {z_std:6.3f} {z_min:6.2f} {max_dz:6.3f}")


if __name__ == "__main__":
    main()
