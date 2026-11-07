#!/usr/bin/env python3
"""Scan dance mocap SOURCES for foot-clean + arm-rich windows (pre-retarget).

Works at the LAFAN-semantic frame level (GMR world: Z-up, meters) so BVH
(LAFAN1, via load_bvh_file) and AMC (CMU 137 family, via
tools_amc_to_lafan_frames FK + alignment) are scored identically.

Per sliding window (default 750 f @30fps = 25 s):
  root stability : z_std<=0.22, z_min>=1.15, max|dz|<=0.06 (six-gate compatible)
  foot cleanliness (contact frames: lowest(ankle,toe) z < 0.05 m):
    sole_pitch = atan2(ankle_z - toe_z, horiz(ankle,toe))
      + = ankle above toe (normal foot has +5..20 deg; tiptoe > 40)
    tiptoe_pct  : pitch > 40 deg among contact frames
    flat_pct    : 0 <= pitch <= 30 deg among contact frames
    toeout_deg  : mean |angle(toe->ankle dir, body forward)| (body forward =
                  perp of hip line, projected to ground); > 35 = duck feet
  arm richness  : mean wrist speed rel to shoulder (m/s) + wrist rel-pos
                  volume (std product) — both sides averaged
  travel_m      : root xy net displacement

Usage:
    .venv/bin/python tools_scan_dance_sources.py --win 750 --stride 150
    .venv/bin/python tools_scan_dance_sources.py --bvh downloads/lafan1/dance1_subject1.bvh
    .venv/bin/python tools_scan_dance_sources.py --asf downloads/cmu137/137.asf --amc downloads/cmu137/137_04.amc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DEFAULT_GMR_ROOT = REPO / "GMR"

BODIES = ["Hips", "LeftUpLeg", "RightUpLeg", "LeftFoot", "LeftToe",
          "RightFoot", "RightToe", "LeftArm", "LeftHand", "RightArm", "RightHand"]


def load_bvh_frames(path: Path):
    sys.path.insert(0, str(DEFAULT_GMR_ROOT))
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    frames, _ = load_bvh_file(str(path), format="lafan1")
    return frames, 30


def load_amc_frames(asf: Path, amc: Path, target_fps: int):
    from tools_amc_to_lafan_frames import (
        align_to_lafan, convert_frames, fk_skeleton, lafan_reference_frame, parse_amc, parse_asf,
    )
    asf = parse_asf(asf)
    amc = parse_amc(amc)
    stride = 120 // target_fps
    src_idx = list(range(0, len(amc), stride))
    # stance = most stationary window (same as tools_amc_csv_f1.py)
    pos = np.array([fk_skeleton(asf, amc[i])["root"][0][:2] for i in src_idx])
    w = max(2, len(pos) // 10)
    moves = np.convolve(np.linalg.norm(np.diff(pos, axis=0), axis=1), np.ones(w), "valid")
    stance = src_idx[min(int(np.argmin(moves)) + w // 2, len(src_idx) - 1)]
    lafan_stance = lafan_reference_frame(DEFAULT_GMR_ROOT,
                                         REPO / "downloads/lafan1/dance1_subject3.bvh")
    R_yaw, B = align_to_lafan(fk_skeleton(asf, amc[stance]), lafan_stance)
    # ground normalization over the whole clip
    from tools_amc_to_lafan_frames import _to_gmr
    zs = []
    for i in src_idx:
        fk = fk_skeleton(asf, amc[i])
        for bone in ("lfoot", "rfoot", "ltoes", "rtoes"):
            zs.append(_to_gmr(*fk[bone])[0][2])
    ground_z = -min(zs)
    return convert_frames(asf, amc, R_yaw, B, src_idx, ground_z=ground_z), target_fps


def frames_to_arrays(frames):
    arrs = {}
    for b in BODIES:
        arrs[b] = np.stack([np.asarray(f[b][0], dtype=float) for f in frames])
    return arrs


def window_metrics(a, s, e, fps):
    root = a["Hips"][s:e]
    travel = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    z = root[:, 2]
    dz = np.abs(np.diff(z))
    # GMR world is ground-normalized (standing Hips ~0.85 m); low band excludes
    # floor work (rolls/sits), std/dz exclude jumps. Calibrated on the verified
    # clean LAFAN windows (dance1_subject3 [0,689): std 0.018, maxdz 0.010).
    z_ok = (z.std() <= 0.22) and (z.min() >= 0.60) and ((dz.max() <= 0.06) if len(dz) else True)

    foot = {}
    for side in ("Left", "Right"):
        ank, toe = a[f"{side}Foot"][s:e], a[f"{side}Toe"][s:e]
        low = np.minimum(ank[:, 2], toe[:, 2])
        ground = low.min()
        contact = low < ground + 0.05
        if contact.sum() < 10:
            foot[side] = dict(contact=0, pitch_med=np.nan, tiptoe_pct=100.0,
                              flat_pct=0.0, toeout=np.nan)
            continue
        az, tz = ank[contact, 2], toe[contact, 2]
        horiz = np.linalg.norm(ank[contact, :2] - toe[contact, :2], axis=1).clip(1e-4)
        pitch = np.degrees(np.arctan2(az - tz, horiz))
        # body forward: perp of hip line (right->left), Z-up world
        lh, rh = a["LeftUpLeg"][s:e], a["RightUpLeg"][s:e]
        hipvec = np.mean((lh - rh)[contact], axis=0)
        fwd = np.array([hipvec[1], -hipvec[0], 0.0])  # rotate hip line by -90 deg
        fwd[2] = 0.0
        fwd /= max(np.linalg.norm(fwd[:2]), 1e-6)
        tdir = toe[contact, :2] - ank[contact, :2]
        tdir /= np.maximum(np.linalg.norm(tdir, axis=1, keepdims=True), 1e-6)
        toeout = float(np.degrees(np.mean(np.abs(np.arccos(np.clip(tdir @ fwd[:2], -1, 1))))))
        foot[side] = dict(contact=int(contact.sum()), pitch_med=float(np.median(pitch)),
                          tiptoe_pct=float((pitch > 40).mean() * 100),
                          flat_pct=float(((pitch >= 0) & (pitch <= 30)).mean() * 100),
                          toeout=toeout)

    arm = {}
    for side in ("Left", "Right"):
        rel = a[f"{side}Hand"][s:e] - a[f"{side}Arm"][s:e]
        speed = float(np.mean(np.linalg.norm(np.diff(rel, axis=0), axis=1)) * fps)
        vol = float(np.prod(rel.std(axis=0)))
        arm[side] = dict(speed=speed, vol=vol)
    arm_speed = (arm["Left"]["speed"] + arm["Right"]["speed"]) / 2
    arm_vol = (arm["Left"]["vol"] + arm["Right"]["vol"]) / 2
    fc = [foot[s] for s in ("Left", "Right")]
    okc = [f for f in fc if f["contact"] > 0]
    pitch_med = float(np.mean([f["pitch_med"] for f in okc])) if okc else np.nan
    tiptoe = float(np.mean([f["tiptoe_pct"] for f in okc])) if okc else 100.0
    flat = float(np.mean([f["flat_pct"] for f in okc])) if okc else 0.0
    toeout = float(np.mean([f["toeout"] for f in okc if f["toeout"] == f["toeout"]])) \
        if any(f["toeout"] == f["toeout"] for f in okc) else np.nan
    return dict(z_ok=bool(z_ok), travel=travel, pitch_med=pitch_med, tiptoe_pct=tiptoe,
                flat_pct=flat, toeout=toeout, arm_speed=arm_speed, arm_vol=arm_vol,
                L=foot["Left"], R=foot["Right"])


def scan(name, frames, fps, win, stride, min_window):
    a = frames_to_arrays(frames)
    n = len(a["Hips"])
    rows = []
    for s in range(0, max(n - win + 1, 1), stride):
        e = min(s + win, n)
        if e - s < min_window:
            break
        m = window_metrics(a, s, e, fps)
        m["start"], m["end"], m["name"] = s, e, name
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=750)
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--min_window", type=int, default=600)
    ap.add_argument("--bvh", type=Path, nargs="*", default=None)
    ap.add_argument("--asf", type=Path, default=None)
    ap.add_argument("--amc", type=Path, nargs="*", default=None)
    ap.add_argument("--json_out", type=Path, default=None)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    bvh_files = args.bvh
    if not bvh_files and not args.amc:
        bvh_files = sorted((REPO / "downloads/lafan1").glob("dance*_subject*.bvh"))

    all_rows = []
    if bvh_files:
        for f in bvh_files:
            frames, fps = load_bvh_frames(f)
            all_rows += scan(f.stem, frames, fps, args.win, args.stride, args.min_window)
            print(f"[scan] {f.name}: {len(frames)} frames", file=sys.stderr)
    if args.amc:
        for f in args.amc:
            frames, fps = load_amc_frames(args.asf, f, 30)
            all_rows += scan(f.stem, frames, fps, args.win, args.stride, args.min_window)
            print(f"[scan] {f.name}: {len(frames)} frames", file=sys.stderr)

    # composite: foot-clean dominates, arm richness breaks ties
    def score(r):
        foot = r["flat_pct"] - 0.5 * r["tiptoe_pct"]
        arm = min(r["arm_speed"], 1.5)  # cap: beyond 1.5 m/s no extra credit
        return (foot if r["z_ok"] else foot - 100) + arm

    all_rows.sort(key=score, reverse=True)
    hdr = (f"{'source':26s} {'start':>5s} {'end':>5s} {'zok':>3s} {'trav':>5s} "
           f"{'pitch':>5s} {'tipt%':>5s} {'flat%':>5s} {'toeout':>6s} "
           f"{'armV':>5s} {'armVol':>7s}")
    print(hdr)
    for r in all_rows[:args.top]:
        print(f"{r['name']:26s} {r['start']:5d} {r['end']:5d} "
              f"{'Y' if r['z_ok'] else 'N':>3s} {r['travel']:5.2f} "
              f"{r['pitch_med']:5.1f} {r['tiptoe_pct']:5.1f} {r['flat_pct']:5.1f} "
              f"{r['toeout']:6.1f} {r['arm_speed']:5.2f} {r['arm_vol']:7.4f}")
    if args.json_out:
        import json
        args.json_out.write_text(json.dumps(all_rows, indent=2, default=str))
        print(f"json -> {args.json_out}")


if __name__ == "__main__":
    main()
