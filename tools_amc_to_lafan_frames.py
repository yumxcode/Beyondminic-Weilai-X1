#!/usr/bin/env python3
"""CMU ASF/AMC -> GMR LAFAN-semantic frames -> F1 retarget CSV.

Converts a CMU mocap trial (subject 137 family: stylistic walks with
whole-body travel) into the exact frame format produced by
GMR utils/lafan1.py::load_bvh_file, so the F1 retargeting pipeline
(tools_bvh_to_csv_f1.py IK loop + quality gates) is reused verbatim.

Frame format (GMR world: Y-up, meters, quat wxyz):
    { "Hips": [pos(3,), quat(4,)], ... }  # 15 LAFAN bodies + FootMod

Pipeline:
  1. parse ASF skeleton (bonedata direction/length/axis/dof, hierarchy)
  2. parse AMC frames; FK per frame
       R_local = R_axis * R_dof;  world_R = parent_world_R @ R_local
       pos[i]   = pos[parent] + parent_world_R @ (dir_unit * length * S)
     S = 2.54/0.45/100 (ASF unit -> m, verified by T-pose body size check)
     root translation: inches -> m (verified by walking root height)
  3. skeleton-convention alignment to LAFAN:
     - R_yaw: rotate CMU world so hips yaw matches the LAFAN reference
     - B_i = inv(R_yaw * q_cmu_stand_i) @ q_lafan_stand_i   (per body)
     - output quat_i(t) = R_yaw * q_cmu_i(t) * B_i
     (reference stance: LAFAN dance1_subject3 frame 0 vs CMU stance frame)
  4. emit LAFAN-named frames incl. FootMod (foot pos + toes rot)

Self-checks (--check): T-pose layout, stance root height, foot ground
contact during walk, alignment residual per body.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rt

REPO = Path(__file__).resolve().parent
DEFAULT_GMR_ROOT = REPO / "GMR"

# CMU bone -> LAFAN body (position source == rotation source unless FootMod)
CMU_TO_LAFAN = {
    "root": "Hips",
    "lfemur": "LeftUpLeg", "ltibia": "LeftLeg", "lfoot": "LeftFoot", "ltoes": "LeftToe",
    "rfemur": "RightUpLeg", "rtibia": "RightLeg", "rfoot": "RightFoot", "rtoes": "RightToe",
    "lowerback": "Spine", "upperback": "Spine1", "thorax": "Spine2",
    "lhumerus": "LeftArm", "lradius": "LeftForeArm", "lhand": "LeftHand",
    "rhumerus": "RightArm", "rradius": "RightForeArm", "rhand": "RightHand",
}

ASF_LEN_SCALE = 2.54 / 0.45 / 100.0  # ASF length unit -> meters
ROOT_TRANS_SCALE = 2.54 / 0.45 / 100.0  # AMC root translation uses ASF units too
                                           # (verified: walking root height ~0.9 m)


def parse_asf(path: Path) -> dict:
    text = path.read_text()
    root_m = re.search(r":root(.*?):bonedata", text, re.S).group(1)
    root_order = re.search(r"order\s+([A-Z\s]+)", root_m).group(1).split()
    root_axis = re.search(r"axis\s+([XYZ]+)", root_m).group(1)

    bones = {}
    for seg in re.findall(r"begin(.*?)end", text, re.S):
        name = re.search(r"name\s+(\S+)", seg)
        if not name:
            continue
        name = name.group(1)
        d = re.search(r"direction\s+([-\d.e\s+]+)", seg).group(1).split()
        direction = np.array([float(v) for v in d[:3]])
        length = float(re.search(r"length\s+([\d.e+-]+)", seg).group(1))
        axis_m = re.search(r"axis\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([XYZ]+)", seg)
        axis = np.array([float(axis_m.group(i)) for i in (1, 2, 3)])
        axis_order = axis_m.group(4).lower()
        dof_m = re.search(r"dof\s+([a-z\s]+)", seg)
        dof = dof_m.group(1).split() if dof_m else []
        bones[name] = dict(direction=direction / np.linalg.norm(direction),
                           length=length, axis=axis, axis_order=axis_order, dof=dof)

    hier_text = re.search(r":hierarchy(.*?)$", text, re.S).group(1)
    hier_text = hier_text.split("end")[0]
    parent = {}
    for line in hier_text.strip().splitlines():
        toks = line.split()
        if len(toks) >= 2:
            for child in toks[1:]:
                parent[child] = toks[0]
    return dict(bones=bones, parent=parent, root_order=root_order, root_axis=root_axis)


def parse_amc(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    frames, cur = [], None
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(":"):
            continue
        toks = s.split()
        if re.fullmatch(r"\d+", toks[0]):
            if cur:
                frames.append(cur)
            cur = {"root": []}
            continue
        if cur is None:
            continue
        vals = [float(v) for v in toks[1:]]
        cur[toks[0]] = vals
    if cur:
        frames.append(cur)
    return frames


def _euler(rot_order: str, angles_deg) -> Rt:
    return Rt.from_euler(rot_order, np.asarray(angles_deg, dtype=float), degrees=True)


def fk_skeleton(asf: dict, pose: dict):
    """pose: {'root': [tx ty tz rx ry rz], bone: [dof values...]} -> {bone: (pos, R)}"""
    bones, parent = asf["bones"], asf["parent"]
    tx, ty, tz = [v * ROOT_TRANS_SCALE for v in pose["root"][:3]]
    rroot = _euler("xyz" if asf["root_axis"].lower() == "xyz" else asf["root_axis"].lower(),
                   pose["root"][3:6])
    out = {"root": (np.array([tx, ty, tz]), rroot)}
    # BFS over hierarchy
    children = {}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)
    queue = list(children.get("root", []))
    while queue:
        name = queue.pop(0)
        par = parent[name]
        if par not in out:
            queue.append(name)
            continue
        b = bones[name]
        p_par, R_par = out[par]
        R_axis = _euler(b["axis_order"], b["axis"])
        vals = pose.get(name)
        if vals is None:  # missing bone in this frame -> rest dof
            vals = [0.0] * len(b["dof"])
        dof_map = dict(zip(b["dof"], vals))
        R_dof = _euler("xyz", [dof_map.get("rx", 0.0), dof_map.get("ry", 0.0), dof_map.get("rz", 0.0)]) \
            if b["dof"] else Rt.identity()
        p_par, R_par = out[par]
        R_local = R_axis * R_dof
        R_world = R_par * R_local
        p_world = p_par + R_par.apply(b["direction"] * b["length"] * ASF_LEN_SCALE)
        out[name] = (p_world, R_world)
        queue.extend(children.get(name, []))
    return out


def lafan_reference_frame(gmr_root: Path, bvh: Path, frame_idx: int = 0):
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    frames, _ = load_bvh_file(str(bvh), format="lafan1")
    return frames[frame_idx]


# CMU world (Y-up) -> GMR world (Z-up, forward +z, left +x):
# (x,y,z) -> (x,-z,y)  == Rx(-90 deg); matches load_bvh_file's rotation_matrix
CMU_TO_GMR = Rt.from_euler("x", -90, degrees=True)


def _to_gmr(p: np.ndarray, R: Rt) -> tuple[np.ndarray, Rt]:
    return np.array([p[0], -p[2], p[1]]), CMU_TO_GMR * R


def align_to_lafan(cmu_stance: dict, lafan_stance: dict):
    """Returns (R_yaw, B_i dict) mapping CMU-FK world -> GMR/LAFAN world."""
    q_hips_l = Rt.from_quat(np.array(lafan_stance["Hips"][1]), scalar_first=True)
    _, q_hips_c = _to_gmr(*cmu_stance["root"])
    # heading alignment in the GMR xy-plane. The hips local +x is near-vertical
    # (degenerate), so we disambiguate 180-degree candidates by voting on
    # per-body stance positions (both stances are natural standing poses).
    def heading(q):
        f = q.apply([1.0, 0.0, 0.0])
        return np.arctan2(f[1], f[0])

    def stance_err(yaw_val):
        R = Rt.from_euler("z", yaw_val)
        hips_c0, _ = _to_gmr(*cmu_stance["root"])
        hips_l0 = np.array(lafan_stance["Hips"][0])
        err = 0.0
        n = 0
        for cmu_name, laf_name in CMU_TO_LAFAN.items():
            if laf_name not in lafan_stance or cmu_name not in cmu_stance:
                continue
            p, _ = _to_gmr(*cmu_stance[cmu_name])
            p = R.apply(p) - R.apply(hips_c0)
            err += np.linalg.norm(p - (np.array(lafan_stance[laf_name][0]) - hips_l0))
            n += 1
        return err / max(n, 1)

    yaw0 = heading(q_hips_l) - heading(q_hips_c)
    cands = (yaw0, yaw0 + np.pi)
    errs = [stance_err(y) for y in cands]
    yaw = cands[int(np.argmin(errs))]
    print(f"[align] yaw candidates {[round(np.degrees(c),1) for c in cands]} deg, "
          f"stance pos err {[round(e,3) for e in errs]} m -> chose {round(np.degrees(yaw),1)} deg")
    R_yaw = Rt.from_euler("z", yaw)
    B = {}
    for cmu_name, laf_name in CMU_TO_LAFAN.items():
        if laf_name not in lafan_stance or cmu_name not in cmu_stance:
            continue
        _, q_c = _to_gmr(*cmu_stance[cmu_name])
        q_c = R_yaw * q_c
        q_l = Rt.from_quat(np.array(lafan_stance[laf_name][1]), scalar_first=True)
        B[laf_name] = q_c.inv() * q_l
    return R_yaw, B


def flatten_yaw_only(q: Rt) -> Rt:
    """Level a bone orientation: keep its yaw, force up to world +Z.

    Runway-walk sources have strong plantar flexion (pointed toes); a flat
    robot foot that copies it digs a corner into the ground. Keep only the
    heading: rebuild R from the horizontal projection of the bone's local
    +Z (CMU foot/toe bones point along local +Z)."""
    f = q.apply(np.array([0.0, 0.0, 1.0]))
    f_xy = f[:2]
    n = np.linalg.norm(f_xy)
    if n < 1e-6:  # bone pointing straight up/down -> keep original
        return q
    f_h = np.array([f_xy[0] / n, f_xy[1] / n, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(up, f_h)  # right-handed: side = up x forward
    R = np.column_stack([f_h, side, up])
    return Rt.from_matrix(R)


def convert_frames(asf, amc_frames, R_yaw, B, src_idx, ground_z=0.0):
    frames = []
    for idx in src_idx:
        pose = amc_frames[idx]
        fk = fk_skeleton(asf, pose)
        result = {}
        for cmu_name, laf_name in CMU_TO_LAFAN.items():
            if cmu_name not in fk:
                continue
            p, R = _to_gmr(*fk[cmu_name])
            p_out = R_yaw.apply(p)
            p_out[2] += ground_z  # normalize the CMU world ground to z=0
            q_out = (R_yaw * R * B.get(laf_name, Rt.identity()))
            if laf_name in ("LeftFoot", "RightFoot", "LeftToe", "RightToe"):
                # runway walks have strong plantar flexion (pointed toes);
                # a flat robot foot copying it digs into the ground — keep
                # only the heading (yaw) of the foot/toe orientation
                q_out = flatten_yaw_only(q_out)
            result[laf_name] = [p_out.astype(np.float64), q_out.as_quat(scalar_first=True)]
        if "LeftFoot" in result and "LeftToe" in result:
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        frames.append(result)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asf", type=Path, required=True)
    ap.add_argument("--amc", type=Path, required=True)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=None)
    ap.add_argument("--stance_frame", type=int, default=-1,
                    help="CMU frame index used for skeleton alignment (a standing pose); -1 = auto")
    ap.add_argument("--target_fps", type=int, default=30,
                    help="AMC is 120 fps; frames are downsampled to this rate")
    ap.add_argument("--ref_bvh", type=Path, default=REPO / "downloads/lafan1/dance1_subject3.bvh")
    ap.add_argument("--check", action="store_true", help="run self-checks and exit")
    args = ap.parse_args()

    asf = parse_asf(args.asf)
    amc = parse_amc(args.amc)
    print(f"[asf] {len(asf['bones'])} bones; [amc] {len(amc)} frames")

    if args.check:
        # 1. T-pose (all dof zero) layout
        zero_pose = {"root": [0, 0, 0, 0, 0, 0]}
        for name, b in asf["bones"].items():
            zero_pose[name] = [0.0] * len(b["dof"])
        fk = fk_skeleton(asf, zero_pose)
        names = ["root", "lowerback", "upperback", "thorax", "head",
                 "lhumerus", "lradius", "lhand", "rhumerus", "rradius",
                 "lfemur", "ltibia", "lfoot", "ltoes", "rfemur", "rtibia", "rfoot", "rtoes"]
        print("== T-pose layout (x,y,z m) ==")
        for n in names:
            if n in fk:
                p = fk[n][0]
                print(f"  {n:12s} {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}")
        # 2. stance / walk stats
        root_z, foot_z = [], []
        for i in range(0, min(len(amc), 400)):
            fk = fk_skeleton(asf, amc[i])
            root_z.append(fk["root"][0][1])
            foot_z.append(min(fk["lfoot"][0][1], fk["rfoot"][0][1]))
        root_z, foot_z = np.array(root_z), np.array(foot_z)
        print(f"== first 400 frames: root_y {root_z.min():.3f}..{root_z.max():.3f} "
              f"(walking ~0.9), foot_y {foot_z.min():.3f}..{foot_z.max():.3f} (ground ~0) ==")
        return

    lafan_stance = lafan_reference_frame(DEFAULT_GMR_ROOT, args.ref_bvh)
    if args.stance_frame < 0:  # auto: most stationary 120-frame window
        stride = 120 // args.target_fps
        idxs = list(range(0, len(amc), stride))
        pos = np.array([fk_skeleton(asf, amc[i])["root"][0][:2] for i in idxs])
        w = max(2, len(pos) // 10)
        moves = np.convolve(np.linalg.norm(np.diff(pos, axis=0), axis=1),
                            np.ones(w), "valid")
        stance_out_idx = int(np.argmin(moves)) + w // 2
        args.stance_frame = idxs[min(stance_out_idx, len(idxs) - 1)]
        print(f"[align] auto stance frame {args.stance_frame} (@{args.target_fps}fps idx {stance_out_idx})")
    cmu_stance = fk_skeleton(asf, amc[args.stance_frame])
    R_yaw, B = align_to_lafan(cmu_stance, lafan_stance)
    # alignment residual on stance frame (sanity, degrees)
    conv = convert_frames(asf, amc, R_yaw, B, [args.stance_frame])[0]
    print("== alignment residual (stance, deg | pos m) ==")
    for laf in sorted(B):
        q1 = Rt.from_quat(conv[laf][1], scalar_first=True)
        q2 = Rt.from_quat(np.array(lafan_stance[laf][1]), scalar_first=True)
        d_deg = np.degrees((q1.inv() * q2).magnitude())
        d_pos = np.linalg.norm(conv[laf][0] - np.array(lafan_stance[laf][0]))
        print(f"  {laf:14s} rot {d_deg:6.1f}  pos {d_pos:.3f}")

    end = len(amc) if args.end_frame is None else args.end_frame
    stride = 120 // args.target_fps
    src_idx = list(range(args.start_frame, end, stride))
    frames = convert_frames(asf, amc, R_yaw, B, src_idx, src_idx)
    out = args.amc.with_suffix(".lafan_frames.npz")
    # compact save: per-body arrays
    arrs = {}
    for laf in set().union(*[set(f) for f in frames[:1]]):
        arrs[f"{laf}_pos"] = np.stack([f[laf][0] for f in frames])
        arrs[f"{laf}_quat"] = np.stack([f[laf][1] for f in frames])
    np.savez(out, **arrs)
    travel = np.linalg.norm(frames[-1]["Hips"][0][:2] - frames[0]["Hips"][0][:2])
    print(f"[out] {len(frames)} frames -> {out.name}; Hips travel {travel:.2f} m")


if __name__ == "__main__":
    main()
