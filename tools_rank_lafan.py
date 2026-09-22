"""Rank LAFAN1 dance BVH files by retargeting difficulty for X1.

Metrics per BVH: min/max hips height (crouch/jump), max foot speed,
max hips rotation speed, fraction of frames with hips below 0.55 m.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "GMR"))

from general_motion_retargeting.utils.lafan1 import load_bvh_file

files = sorted((REPO / "downloads/lafan1").glob("dance*.bvh"))
print(f"{'file':28s} {'frames':>6s} {'minHip':>7s} {'maxHip':>7s} {'maxFootV':>8s} {'lowHip%':>7s}")
for f in files:
    try:
        frames, _ = load_bvh_file(str(f), format="lafan1")
    except Exception as e:
        print(f"{f.name:28s} ERROR {e}")
        continue
    n = len(frames)
    hips = np.array([fr["Hips"][0] for fr in frames])
    lf = np.array([fr["LeftFootMod"][0] for fr in frames])
    rf = np.array([fr["RightFootMod"][0] for fr in frames])
    foot_v = max(np.abs(np.diff(lf, axis=0)).max(), np.abs(np.diff(rf, axis=0)).max()) * 30
    low = (hips[:, 2] < 0.55).mean() * 100
    print(f"{f.name:28s} {n:6d} {hips[:,2].min():7.3f} {hips[:,2].max():7.3f} {foot_v:8.2f} {low:6.1f}%")
