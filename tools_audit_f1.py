#!/usr/bin/env python3
"""F1_29DOF asset audit.

Audits F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf:
  1. tree structure & joint table (revolute joints: axis/limits/effort/velocity/damping)
  2. mass & inertia sanity (positive-definite, total mass, subtree masses)
  3. mesh reference existence
  4. leg kinematic dimensions (hip->knee->ankle segment lengths, foot sole height
     from ankle_roll frame via collision mesh AABB in MuJoCo)
  5. per-joint subtree reflected inertia about the joint axis (kp/kd sizing input)
  6. standing keypose: FK at zero joints -> sole height & base height for standing
  7. MuJoCo compile check of the raw URDF

Outputs a machine-readable report to stdout.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
F1_URDF = REPO / "F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"
X1_URDF = REPO / "X1_29DOF/urdf/f1.urdf"


def parse_urdf(path: Path):
    root = ET.parse(path).getroot()
    links = {}
    joints = []
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value")) if inertial is not None else 0.0
        links[name] = {"mass": mass, "inertial": inertial}
    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split())] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split())] if origin is not None else [0, 0, 0]
        axis_el = joint.find("axis")
        axis = [float(v) for v in axis_el.get("xyz").split()] if axis_el is not None else [0, 0, 1]
        limit = joint.find("limit")
        lim = None
        if limit is not None:
            lim = {
                "lower": float(limit.get("lower")),
                "upper": float(limit.get("upper")),
                "effort": float(limit.get("effort")),
                "velocity": float(limit.get("velocity")),
            }
        dyn = joint.find("dynamics")
        damping = float(dyn.get("damping")) if dyn is not None and dyn.get("damping") else 0.0
        friction = float(dyn.get("friction")) if dyn is not None and dyn.get("friction") else 0.0
        joints.append({
            "name": name, "type": jtype, "parent": parent, "child": child,
            "xyz": xyz, "rpy": rpy, "axis": axis, "limit": lim,
            "damping": damping, "friction": friction,
        })
    return links, joints


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def main():
    links, joints = parse_urdf(F1_URDF)
    x1_links, x1_joints = parse_urdf(X1_URDF)
    x1_joint_by_name = {j["name"]: j for j in x1_joints}

    rev = [j for j in joints if j["type"] == "revolute"]
    fixed = [j for j in joints if j["type"] == "fixed"]
    print(f"== F1 URDF audit: {F1_URDF.name} ==")
    print(f"links={len(links)} joints={len(joints)} revolute={len(rev)} fixed={len(fixed)}")
    total_mass = sum(l["mass"] for l in links.values())
    print(f"total_mass={total_mass:.3f} kg")

    # --- joint table + X1 comparison ---
    print("\n== revolute joints (F1 | X1-diff) ==")
    print(f"{'joint':34s} {'axis':15s} {'lo':>7s} {'hi':>7s} {'eff':>7s} {'vel':>7s} {'damp':>5s} {'fric':>5s} | x1: axis/lo/hi/eff")
    for j in sorted(rev, key=lambda x: x["name"]):
        lim = j["limit"]
        diff = ""
        x1j = x1_joint_by_name.get(j["name"])
        if x1j is not None:
            x1l = x1j["limit"] or {}
            marks = []
            if not np.allclose(j["axis"], x1j["axis"]):
                marks.append(f"axis {x1j['axis']}")
            for k, tag in (("lower", "lo"), ("upper", "hi"), ("effort", "eff")):
                if k in x1l and abs(lim[k] - x1l[k]) > 1e-6:
                    marks.append(f"{tag} {x1l[k]:g}")
            diff = "; ".join(marks) if marks else "same"
        else:
            diff = "NO X1 COUNTERPART"
        print(f"{j['name']:34s} {str(j['axis']):15s} {lim['lower']:7.3f} {lim['upper']:7.3f} "
              f"{lim['effort']:7.1f} {lim['velocity']:7.2f} {j['damping']:5.2f} {j['friction']:5.2f} | {diff}")

    # missing/extra vs X1
    f1_names = {j["name"] for j in rev}
    x1_names = {j["name"] for j in x1_joints if j["type"] == "revolute"}
    print(f"\nF1-only revolute: {sorted(f1_names - x1_names)}")
    print(f"X1-only revolute: {sorted(x1_names - f1_names)}")

    # --- inertia sanity ---
    print("\n== inertia sanity ==")
    bad = []
    for name, l in links.items():
        ine = l["inertial"]
        if ine is None:
            continue
        I = ine.find("inertia")
        ixx, iyy, izz = (float(I.get(k)) for k in ("ixx", "iyy", "izz"))
        if ixx <= 0 or iyy <= 0 or izz <= 0:
            bad.append(f"{name}: non-positive diag {ixx},{iyy},{izz}")
        # triangle inequality
        if ixx + iyy < izz - 1e-9 or iyy + izz < ixx - 1e-9 or izz + ixx < iyy - 1e-9:
            bad.append(f"{name}: triangle inequality violated")
    print("OK" if not bad else "\n".join(bad))

    # --- mesh existence ---
    print("\n== mesh references ==")
    missing = []
    for mesh in root_meshes(F1_URDF):
        if not (F1_URDF.parent.parent / mesh).exists():
            missing.append(mesh)
    print(f"all mesh files exist: {not missing}" + (f" missing={missing}" if missing else ""))

    # --- FK: zero-pose geometry ---
    print("\n== zero-pose FK ==")
    # build parent chain transforms
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    world = {"base_link": (np.zeros(3), np.eye(3))}

    def fk(link):
        if link in world:
            return world[link]
        # find joint with child == link
        pj = next(j for j in joints if j["child"] == link)
        pp, pR = fk(pj["parent"])
        R = pR @ rpy_to_R(pj["rpy"])
        p = pp + pR @ np.array(pj["xyz"])
        world[link] = (p, R)
        return world[link]

    for name in ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee_pitch",
                 "left_ankle_pitch", "left_ankle_roll", "lumbar_yaw_link",
                 "left_shoulder_pitch_link", "left_elbow_pitch_link", "left_wrist_roll_link"]:
        p, _ = fk(name)
        print(f"{name:28s} pos={np.round(p, 4)}")

    # segment lengths
    hip_pitch, _ = fk("left_hip_pitch")
    hip_yaw, _ = fk("left_hip_yaw")
    knee, _ = fk("left_knee_pitch")
    ankle_p, _ = fk("left_ankle_pitch")
    ankle_r, _ = fk("left_ankle_roll")
    print(f"\nthigh (hip_yaw->knee)   = {np.linalg.norm(knee - hip_yaw):.4f} m")
    print(f"shank (knee->ankle_p)   = {np.linalg.norm(ankle_p - knee):.4f} m")
    print(f"hip->ankle (zero pose)  = {np.linalg.norm(ankle_p - hip_pitch):.4f} m")
    print(f"hip separation (y)      = {2 * abs(fk('right_hip_pitch')[0][1]):.4f} m")

    # --- subtree masses & reflected inertia about joint axis ---
    print("\n== subtree mass & approximate reflected inertia (URDF only, no rotor) ==")
    child_link_of = {j["name"]: j["child"] for j in joints}
    parent_joint_of_link = {j["child"]: j for j in joints}
    link_children = {}
    for j in joints:
        link_children.setdefault(j["parent"], []).append(j["child"])

    def subtree_links(link):
        out = [link]
        for c in link_children.get(link, []):
            out += subtree_links(c)
        return out

    def subtree_mass(link):
        return sum(links[l]["mass"] for l in subtree_links(link))

    # full-body COM per subtree for inertia estimate: I_axis ≈ sum m_i * r_i^2 (+ own I)
    def inertia_about_joint(jname):
        j = next(jj for jj in joints if jj["name"] == jname)
        child = j["child"]
        R_joint_world = fk(child)[1]  # orientation of child frame at zero pose
        axis_world = R_joint_world @ np.array(j["axis"])
        # joint origin in world
        jpos = fk(child)[0]
        I = 0.0
        for l in subtree_links(child):
            ine = links[l]["inertial"]
            if ine is None:
                continue
            m = links[l]["mass"]
            c_local = np.array([float(v) for v in ine.find("origin").get("xyz", "0 0 0").split()])
            lw = fk(l)
            com_w = lw[0] + lw[1] @ c_local
            r = com_w - jpos
            r_perp = r - np.dot(r, axis_world) * axis_world
            I += m * np.dot(r_perp, r_perp)
            # add own inertia projected on axis (rough): use izz-ish via rotation
            Iel = ine.find("inertia")
            diag = np.diag([float(Iel.get("ixx")), float(Iel.get("iyy")), float(Iel.get("izz"))])
            Ia_world = lw[1] @ diag @ lw[1].T
            I += axis_world @ Ia_world @ axis_world
        return I

    print(f"{'joint':34s} {'subtree_m':>9s} {'I_axis':>9s} {'torque_grav':>11s}")
    for j in sorted(rev, key=lambda x: x["name"]):
        child = j["child"]
        sm = subtree_mass(child)
        Ia = inertia_about_joint(j["name"])
        # gravity torque: subtree com offset along -z axis of joint rotation (worst case)
        print(f"{j['name']:34s} {sm:9.3f} {Ia:9.4f}")

    print(f"\nupper_body_mass (lumbar_yaw subtree) = {subtree_mass('lumbar_yaw_link'):.3f} kg")
    print(f"total = {total_mass:.3f} kg; legs+pelvis = {total_mass - subtree_mass('lumbar_yaw_link'):.3f} kg")

    # --- MuJoCo compile check ---
    print("\n== MuJoCo compile check ==")
    try:
        m = mujoco.MjModel.from_xml_path(str(F1_URDF))
        print(f"compiled: nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} nmesh={m.nmesh}")
        # sole height: min z of ankle_roll collision geoms in zero pose
        mujoco.mj_kinematics(m, mujoco.MjData(m))
        for side in ("left", "right"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll")
            zmin = None
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    pos = m.geom_xpos[g]
                    zmin_local = None
                    vadr, vcnt = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                    verts = m.mesh_vert[vadr:vadr + vcnt] @ m.geom_quat[g][[1, 2, 3, 0]]  # not exact; use xpos+R
                    R = np.zeros(9)
                    mujoco.mju_quat2Mat(R, m.geom_quat[g])
                    R = R.reshape(3, 3)
                    verts_w = (R @ m.mesh_vert[vadr:vadr + vcnt].T).T + pos
                    zmin_local = verts_w[:, 2].min()
                    zmin = zmin_local if zmin is None else min(zmin, zmin_local)
            print(f"{side}_ankle_roll sole z_min at zero pose = {zmin:.4f} m")
    except Exception as e:
        print(f"COMPILE FAILED: {e}")


def root_meshes(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    out = []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            out.append(fn.replace("../", ""))
    return sorted(set(out))


if __name__ == "__main__":
    main()
