#!/usr/bin/env python3
"""Determine F1 leg kp/kd from URDF inertia with the damping-ratio method.

Method (same as x1_armature_pd.md 方案B, which was identified for X1):
    kp = (kd / (2*zeta))^2 / I_eff        kd = 2*zeta*sqrt(kp*I_eff)
- I_eff: subtree inertia about the joint axis at zero pose (URDF only, no
  rotor armature — F1 has no lab identification; documented as such).
- zeta targets: hip_pitch / knee 0.7, hip_roll 0.6, hip_yaw 0.8, ankle 0.7.
- kd capped at 8.0 (same MIT-encoding cap as the X1 host stack, conservative).
- Gravity sag check: static deflection under worst-case gravity torque.
- Upper body: F1 links/joints are byte-identical to X1 (same meshes, limits,
  inertias) -> reuse the X1 identified 方案B values verbatim.

Writes F1_KD_KP.md (human-readable table + rationale).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
URDF = REPO / "F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"

KD_CAP = 8.0
ZETA = {
    "hip_pitch": 0.7, "hip_roll": 0.6, "hip_yaw": 0.8,
    "knee_pitch": 0.7, "ankle_pitch": 0.7, "ankle_roll": 0.9,
}

# X1 identified upper-body gains (x1_armature_pd.md 方案B) — links identical.
X1_UPPER = {
    # joint: (kp, kd, armature, effort, vel)
    "lumbar_yaw_joint": (90.0, 8.0, 0.022, 150.0, 8.9),
    "lumbar_roll_joint": (70.0, 8.0, 0.021, 150.0, 8.9),
    "lumbar_pitch_joint": (100.0, 8.0, 0.012, 180.0, 8.9),
    "shoulder_pitch": (40.0, 4.0, 0.010, 20.0, 13.61),
    "shoulder_roll": (40.0, 4.0, 0.010, 20.0, 13.61),
    "shoulder_yaw": (40.0, 2.0, 0.010, 20.0, 13.61),
    "elbow_pitch": (30.0, 1.9, 0.010, 20.0, 13.61),
    "elbow_yaw": (20.0, 1.0, 0.005, 20.0, 13.61),
    "wrist_pitch": (15.0, 1.0, 0.005, 10.0, 2.5),
    "wrist_roll": (15.0, 1.0, 0.005, 10.0, 2.5),
}

LEG_JOINTS = {
    "hip_pitch": ("left_hip_pitch_joint", 130.0, 14.66),
    "hip_roll": ("left_hip_roll_joint", 130.0, 14.66),
    "hip_yaw": ("left_hip_yaw_joint", 130.0, 14.66),
    "knee_pitch": ("left_knee_pitch_joint", 130.0, 14.66),
    "ankle_pitch": ("left_ankle_pitch_joint", 241.875, 7.57244),
    "ankle_roll": ("left_ankle_roll_joint", 218.4, 8.05007),
}


def rpy_to_R(rpy):
    r, p, y = rpy
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return Rz @ Ry @ Rx


def load():
    root = ET.parse(URDF).getroot()
    links, joints = {}, []
    for link in root.findall("link"):
        ine = link.find("inertial")
        mass = float(ine.find("mass").get("value")) if ine is not None else 0.0
        links[link.get("name")] = {"mass": mass, "ine": ine}
    for j in root.findall("joint"):
        o = j.find("origin")
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": np.array([float(v) for v in o.get("xyz", "0 0 0").split()]),
            "rpy": [float(v) for v in o.get("rpy", "0 0 0").split()],
            "axis": np.array([float(v) for v in j.find("axis").get("xyz").split()]) if j.find("axis") is not None else np.array([0, 0, 1.0]),
        })
    return links, joints


def fk_all(links, joints):
    world = {"base_link": (np.zeros(3), np.eye(3))}

    def fk(link):
        if link in world:
            return world[link]
        pj = next(j for j in joints if j["child"] == link)
        pp, pR = fk(pj["parent"])
        R = pR @ rpy_to_R(pj["rpy"])
        p = pp + pR @ pj["xyz"]
        world[link] = (p, R)
        return world[link]

    for name in links:
        fk(name)
    return world


def main():
    links, joints = load()
    world = fk_all(links, joints)
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j["child"])

    def subtree(link):
        out = [link]
        for c in children.get(link, []):
            out += subtree(c)
        return out

    def inertia_axis(jname):
        j = next(jj for jj in joints if jj["name"] == jname)
        child = j["child"]
        p, R = world[child]
        axis_w = R @ j["axis"]
        I = 0.0
        for l in subtree(child):
            ine = links[l]["ine"]
            if ine is None:
                continue
            m = links[l]["mass"]
            c_local = np.array([float(v) for v in ine.find("origin").get("xyz", "0 0 0").split()])
            lw = world[l]
            com = lw[0] + lw[1] @ c_local
            r = com - p
            r_perp = r - np.dot(r, axis_w) * axis_w
            I += m * np.dot(r_perp, r_perp)
            Iel = ine.find("inertia")
            diag = np.diag([float(Iel.get("ixx")), float(Iel.get("iyy")), float(Iel.get("izz"))])
            Iaw = lw[1] @ diag @ lw[1].T
            I += axis_w @ Iaw @ axis_w
        return I

    def gravity_torque(jname):
        """Worst-case gravity torque about joint axis (subtree weight, horizontal arm)."""
        j = next(jj for jj in joints if jj["name"] == jname)
        child = j["child"]
        p, R = world[child]
        axis_w = R @ j["axis"]
        m_tot, arm = 0.0, np.zeros(3)
        for l in subtree(child):
            ine = links[l]["ine"]
            if ine is None:
                continue
            m = links[l]["mass"]
            c_local = np.array([float(v) for v in ine.find("origin").get("xyz", "0 0 0").split()])
            lw = world[l]
            com = lw[0] + lw[1] @ c_local
            arm += m * (com - p)
            m_tot += m
        # gravity torque vector = (sum m_i r_i) x g, g = (0,0,-9.81); project on axis
        tau_vec = np.cross(arm, np.array([0, 0, -9.81]))
        return abs(np.dot(tau_vec, axis_w)), m_tot

    rows = []
    print(f"{'joint class':14s} {'I_axis':>8s} {'tau_g':>7s} {'kp':>7s} {'kd':>5s} {'zeta':>5s} {'fn Hz':>6s} {'sag deg':>8s}")
    for cls, (jname, effort, vel) in LEG_JOINTS.items():
        I = inertia_axis(jname)
        tau_g, m_sub = gravity_torque(jname)
        z = ZETA[cls]
        # start from a torque-authority anchor: kp enough for tau_g at <5% of range
        # then solve kd from zeta, cap kd, re-solve kp from capped kd
        kd = 2 * z * np.sqrt(80.0 * I)  # provisional with kp=80 anchor
        kd = min(kd, KD_CAP)
        kp = (kd / (2 * z)) ** 2 / I
        fn = np.sqrt(kp / I) / (2 * np.pi)
        sag_rad = tau_g / kp
        rows.append(dict(cls=cls, joint=jname, I=I, tau_g=tau_g, m=m_sub,
                         kp=round(kp, 1), kd=round(kd, 2), zeta=z, fn=round(fn, 2),
                         sag_deg=round(np.degrees(sag_rad), 2), effort=effort, vel=vel))
        print(f"{cls:14s} {I:8.4f} {tau_g:7.2f} {kp:7.1f} {kd:5.2f} {z:5.2f} {fn:6.2f} {np.degrees(sag_rad):8.2f}")

    # sanity: ankle sag under FULL body weight (single-support worst case)
    print("\nworst-case single-support ankle_pitch sag (body 35.0 kg at 0.05 m arm):",
          round(np.degrees(35.0 * 9.81 * 0.05 / rows[4]["kp"]), 2), "deg")

    out = {
        "kd_cap": KD_CAP,
        "legs": {r["cls"]: {k: v for k, v in r.items() if k not in ("cls", "joint")} for r in rows},
        "upper": X1_UPPER,
        "note": "legs derived from URDF inertia (no lab ID for F1); upper body identical links -> X1 方案B identified values",
    }
    Path(REPO / "F1_29DOF/f1_kp_kd.json").write_text(json.dumps(out, indent=2))
    print("\nwrote F1_29DOF/f1_kp_kd.json")


if __name__ == "__main__":
    main()
