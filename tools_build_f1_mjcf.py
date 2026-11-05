#!/usr/bin/env python3
"""Build all F1 simulation assets from the vendor URDF (single source of truth).

Inputs:
  F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf   (vendor, ASCII meshes)
  F1_29DOF/meshes/sim/{visual,collision}/*.stl            (converted by tools_convert_f1_meshes.py)

Outputs:
  F1_29DOF/mjcf/robot/f1/f1_serial.xml     MuJoCo serial model (motors+sensors+contacts)
  F1_29DOF/mjcf/environment/flat.xml       flat scene
  F1_29DOF/mjcf/f1_flat.xml                scene include wrapper
  F1_29DOF/mjcf/robot/f1/f1_isaac.urdf     Isaac URDF (30-link tree, sim meshes)

Design (mirrors the validated X1 recipe in X1_29DOF/mjcf):
  - 30-body tree: base_link + 29 actuated links; fixed decorations dropped
    (MuJoCo would fuse them anyway; keeps body order == PhysX merged order).
  - Contact geoms: base + lumbar_pitch + knee collision meshes, 4 corner
    spheres per foot (sole AABB corners, r=0.002) — X1 recipe.
  - Passive damping/frictionloss from the URDF <dynamics> tags.
  - Actuators: torque motors, ctrlrange = URDF effort limits.
  - Sensors: imu site framequat/gyro/framepos/velocimeter/accelerometer +
    jointpos/jointvel x29 in X1 order (legs -> lumbar -> arms).
  - Keyframe "stand": bent-knee standing pose (solved here, reported).
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
F1 = REPO / "F1_29DOF"
SRC_URDF = F1 / "urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf"
OUT_DIR = F1 / "mjcf/robot/f1"

DROP_LINKS = {
    "waist_motor_a_link", "waist_motor_a_ball_link", "waist_motor_a_loop_link",
    "waist_motor_b_link", "waist_motor_b_ball_link", "waist_motor_b_loop_link",
    "arm_r_wrist_a_ball_link", "arm_r_wrist_motor_a_link", "arm_r_wrist_a_loop_link",
    "arm_r_wrist_b_ball_link", "arm_r_wrist_motor_b_link", "arm_r_wrist_b_loop_link",
    "arm_l_wrist_a_ball_link", "arm_l_wrist_motor_a_link", "arm_l_wrist_a_loop_link",
    "arm_l_wrist_b_ball_link", "arm_l_wrist_motor_b_link", "arm_l_wrist_b_loop_link",
    "neck_motor_base_link", "neck_motor_bracket_link", "head_face_bracket_link",
    "camera_link", "radar_link",
}

# X1 sensor joint order (legs first, then lumbar, then arms) — kept for parity.
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
]

# URDF joint order (definition order) — the PhysX joint order for Isaac.
URDF_JOINT_ORDER = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def rewrite_urdf(meshdir: str) -> str:
    """Rewrite the vendor URDF: flat sim-mesh names (see mesh export step),
    drop decoration links/joints. meshdir is baked into the <mujoco><compiler>."""
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    mj = root.find("mujoco")
    if mj is not None:
        root.remove(mj)
    comp = ET.Element("mujoco")
    ET.SubElement(comp, "compiler", {
        "meshdir": meshdir, "balanceinertia": "true", "discardvisual": "false",
        "fusestatic": "false", "strippath": "false", "angle": "radian",
    })
    root.insert(0, comp)

    def remap(fn: str) -> str:
        p_ = fn.replace("../meshes/", "")
        name = Path(p_).name
        if p_.startswith("x1/"):
            return name
        if p_.startswith("visual/"):
            return name
        if p_.startswith("collision/"):
            return "c_" + name.replace("_collision", "")
        return name

    for mesh in root.iter("mesh"):
        mesh.set("filename", remap(mesh.get("filename")))

    drop_joints = [j for j in root.findall("joint")
                   if j.get("type") == "fixed" and j.find("child").get("link") in DROP_LINKS]
    for j in drop_joints:
        root.remove(j)
    for link in root.findall("link"):
        if link.get("name") in DROP_LINKS:
            root.remove(link)
    return ET.tostring(root, encoding="unicode")


def _geom_zmin(model, data, g):
    """World z of the lowest vertex of mesh geom g (data must be kinematics-updated)."""
    vadr, vcnt = model.mesh_vertadr[model.geom_dataid[g]], model.mesh_vertnum[model.geom_dataid[g]]
    R = data.geom_xmat[g].reshape(3, 3)
    vw = (R @ model.mesh_vert[vadr:vadr + vcnt].T).T + data.geom_xpos[g]
    return float(vw[:, 2].min())


def solve_standing(spec: mujoco.MjSpec) -> tuple[np.ndarray, dict]:
    """Symmetric bent-knee standing pose via least squares.

    Unknowns (7): hip_pitch, knee, ankle_pitch, ankle_roll, hip_yaw,
    hip_roll(frontal leveling), lumbar_pitch. Left/right mirrored identically;
    lumbar yaw/roll and arms zeroed. Targets: feet flat, torso upright, CoM
    over support center, knee bend ~0.45.
    """
    from scipy.optimize import least_squares
    model = spec.compile()
    data = mujoco.MjData(model)

    names = JOINT_ORDER
    idx = {n: i for i, n in enumerate(names)}
    jadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    base_id = model.body("base_link").id if hasattr(model.body, "id") else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    lf = body_id(model, "left_ankle_roll")
    rf = body_id(model, "right_ankle_roll")

    def residuals(x):
        (bx, hp, kn, ap, ar, hy, hr, lp) = x
        q = np.zeros(len(names))
        for side in ("left", "right"):
            q[idx[f"{side}_hip_pitch_joint"]] = hp
            q[idx[f"{side}_knee_pitch_joint"]] = kn
            q[idx[f"{side}_ankle_pitch_joint"]] = ap
            q[idx[f"{side}_ankle_roll_joint"]] = ar
            q[idx[f"{side}_hip_yaw_joint"]] = hy
            q[idx[f"{side}_hip_roll_joint"]] = hr if side == "left" else -hr
        q[idx["lumbar_pitch_joint"]] = lp
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[0] = bx
        for a, v in zip(jadr, q):
            data.qpos[a] = v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        res = []
        for bid in (lf, rf):
            R = data.xmat[bid].reshape(3, 3)
            res += [30.0 * R[2, 0], 30.0 * R[2, 1]]          # feet flat
        Rb = data.xmat[base_id].reshape(3, 3)
        res += [30.0 * Rb[2, 0], 30.0 * Rb[2, 1]]             # base upright (guards hip chain)
        com = data.subtree_com[0]
        center = 0.5 * (data.xpos[lf] + data.xpos[rf])
        res += [60.0 * (com[0] - center[0]), 60.0 * (com[1] - center[1])]  # CoM over support
        res.append(8.0 * (kn - 0.45))                          # knee bend target
        res.append(3.0 * ar)                                   # ankles near level
        res.append(2.0 * lp)                                   # prefer straight lumbar
        return res

    x0 = np.array([0.0, -0.25, 0.45, -0.20, 0.0, 0.0, 0.0, 0.0])
    lb = np.array([-0.15, -1.2, 0.20, -0.45, -0.45, -1.2, -0.30, -0.10])
    ub = np.array([0.15, 1.2, 0.90, 0.45, 0.45, 1.2, 0.30, 0.10])
    sol = least_squares(residuals, x0, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, max_nfev=2000)
    (bx, hp, kn, ap, ar, hy, hr, lp) = sol.x
    q = np.zeros(len(names))
    for side in ("left", "right"):
        q[idx[f"{side}_hip_pitch_joint"]] = hp
        q[idx[f"{side}_knee_pitch_joint"]] = kn
        q[idx[f"{side}_ankle_pitch_joint"]] = ap
        q[idx[f"{side}_ankle_roll_joint"]] = ar
        q[idx[f"{side}_hip_yaw_joint"]] = hy
        q[idx[f"{side}_hip_roll_joint"]] = hr if side == "left" else -hr
    q[idx["lumbar_pitch_joint"]] = lp

    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    for a, v in zip(jadr, q):
        data.qpos[a] = v
    mujoco.mj_kinematics(model, data)
    # sole reference: corner-sphere bottom z of both feet (collision-STL AABB)
    _corners = foot_corner_spheres()
    sole_z = min(
        (data.xmat[bid].reshape(3, 3) @ np.array(c))[2] + data.xpos[bid][2]
        for bid, side in ((lf, "left"), (rf, "right"))
        for c in _corners[side]["bottom"]
    )
    info = {"cost": float(2 * np.sum(np.square(residuals(sol.x)))),
            "base_z": 0.0, "sole_z": sole_z,
            "symmetric_params": {k: float(v) for k, v in zip(
                ["base_x", "hip_pitch", "knee", "ankle_pitch", "ankle_roll", "hip_yaw", "hip_roll", "lumbar_pitch"], sol.x)}}
    return q, info


def body_id(model, name: str) -> int:
    """Resolve body id tolerating the _link suffix (F1 legs lack it)."""
    for cand in (name, f"{name}_link", name.removesuffix("_link")):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
        if bid >= 0:
            return bid
    raise KeyError(name)


def foot_corner_spheres() -> dict:
    """Sole AABB corners per foot in the ankle_roll link frame.

    Uses the vendor COLLISION STLs directly (trimesh), transformed by the URDF
    <collision> origin — independent of how MuJoCo mixes visual/collision geoms.
    """
    import trimesh
    urdf = ET.parse(SRC_URDF).getroot()
    out = {}
    for side in ("left", "right"):
        link = next(l for l in urdf.findall("link") if l.get("name") == f"{side}_ankle_roll")
        pts = []
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            fn = mesh.get("filename").replace("../meshes/", "")
            if "collision" not in fn:
                continue
            scale = [float(v) for v in mesh.get("scale", "1 1 1").split()]
            m = trimesh.load(F1 / "meshes" / fn, process=False)
            v = np.asarray(m.vertices) * np.array(scale)
            o = col.find("origin")
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
            xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
            Rz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0], [np.sin(rpy[2]), np.cos(rpy[2]), 0], [0, 0, 1]])
            Ry = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])], [0, 1, 0], [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
            Rx = np.array([[1, 0, 0], [0, np.cos(rpy[0]), -np.sin(rpy[0])], [0, np.sin(rpy[0]), np.cos(rpy[0])]])
            pts.append((Rz @ Ry @ Rx @ v.T).T + xyz)
        pts = np.vstack(pts)
        zmin = pts[:, 2].min()
        sole = pts[pts[:, 2] < zmin + 0.012]
        xmin, xmax = sole[:, 0].min(), sole[:, 0].max()
        ymin, ymax = sole[:, 1].min(), sole[:, 1].max()
        out[side] = {
            "bottom": [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmin, ymax, zmin), (xmax, ymax, zmin)],
            "top": [(x, y, zmin + 0.004) for (x, y, z) in
                    [(xmin, ymin, 0), (xmax, ymin, 0), (xmin, ymax, 0), (xmax, ymax, 0)]],
        }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (F1 / "mjcf/environment").mkdir(parents=True, exist_ok=True)

    # ---------------- MuJoCo serial model ----------------
    # meshdir is resolved relative to the TOP-LEVEL xml at load time: sources
    # compile from mjcf/, exports are loaded via mjcf/f1_flat.xml -> both see
    # meshdir "../../meshes" == F1_29DOF/meshes.
    MJCF_MESHES = F1 / "mjcf/meshes"
    urdf_text = rewrite_urdf(str(MJCF_MESHES.resolve()))
    src = F1 / "mjcf/_f1_source.urdf"
    src.write_text(urdf_text)
    spec = mujoco.MjSpec.from_file(str(src))

    # hide collision meshes (group 3), keep contact behavior — visual geoms are
    # group 1 after URDF import; collision geoms default to visible group 0
    for g in spec.geoms:
        if g.contype is not None and g.contype != 0 or (g.contype is None):
            if g.group is None or g.group == 0:
                g.group = 3

    # freejoint on base
    base = spec.body("base_link")
    if base is None:
        raise SystemExit("base_link not found")
    if not base.joints:
        base.add_joint(name="floating_base", type=mujoco.mjtJoint.mjJNT_FREE)
    # imu site
    base.add_site(name="imu", pos=[0, 0, 0], size=[0.01, 0.01, 0.01])

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    # foot corner spheres (from vendor collision STL AABB)
    corners = foot_corner_spheres()
    for side in ("left", "right"):
        body = spec.body(f"{side}_ankle_roll") if spec.body(f"{side}_ankle_roll") is not None else spec.body(f"{side}_ankle_roll_link")
        for (x, y, z) in corners[side]["top"]:  # visible markers
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0, 0],
                          pos=[float(x), float(y), float(z)], contype=0, conaffinity=0,
                          group=3, rgba=[0.9, 0.1, 0.1, 0.35])
        ct, ca = (2, 4) if side == "left" else (4, 2)  # X1 collision-left/right convention
        for (x, y, z) in corners[side]["bottom"]:      # contact corners
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0, 0],
                          pos=[float(x), float(y), float(z)], contype=ct, conaffinity=ca,
                          group=3)
        c = corners[side]
        print(f"{side} foot corners (ankle_roll frame): "
              f"x[{c['bottom'][0][0]:.4f},{c['bottom'][1][0]:.4f}] "
              f"y[{c['bottom'][0][1]:.4f},{c['bottom'][2][1]:.4f}] "
              f"z_bottom={c['bottom'][0][2]:.4f}")

    # actuators: torque motors per joint (ctrlrange = effort)
    spec.actuators.clear()
    efforts = {"hip_pitch": 130.0, "hip_roll": 130.0, "hip_yaw": 130.0, "knee_pitch": 130.0,
               "ankle_pitch": 241.875, "ankle_roll": 218.4,
               "lumbar_yaw": 150.0, "lumbar_roll": 150.0, "lumbar_pitch": 180.0,
               "shoulder": 20.0, "elbow": 20.0, "wrist": 10.0}
    for jn in JOINT_ORDER:
        eff = next(v for k, v in efforts.items() if k in jn)
        spec.add_actuator(name=f"motor_{jn}", target=jn,
                        trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-eff, eff], ctrllimited=True)

    # sensors: X1 layout/order
    spec.sensors.clear()
    spec.add_sensor(name="body-orientation", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="body-angular-velocity", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu", noise=0.001)
    spec.add_sensor(name="body-linear-pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="body-linear-vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="body-linear-acceleration", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu", noise=0.001)
    for jn in JOINT_ORDER:
        spec.add_sensor(name=f"jointpos_{jn}", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=jn)
        spec.add_sensor(name=f"jointvel_{jn}", type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=jn)

    # standing keyframe
    q_stand, info = solve_standing(spec)
    print("standing pose solved:", {k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()})
    qpos0 = np.zeros(7 + len(JOINT_ORDER))
    qpos0[3] = 1.0
    qpos0[2] = -info["sole_z"]  # base height so soles touch z=0
    for i, v in enumerate(q_stand):
        qpos0[7 + i] = v
    # qpos joint slots are in spec joint-definition order == URDF order? verify by name mapping
    _m = spec.compile()
    _qadr = {n: _m.jnt_qposadr[mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_ORDER}
    qpos_named = np.zeros(_m.nq)
    qpos_named[3] = 1.0
    qpos_named[0] = float(info["symmetric_params"]["base_x"])
    qpos_named[2] = -info["sole_z"] + 0.002  # sphere centers at r above ground
    for n, v in zip(JOINT_ORDER, q_stand):
        qpos_named[_qadr[n]] = v
    kf = spec.add_key(name="stand")
    kf.qpos = [float(v) for v in qpos_named]
    spec.compile()  # to_file only writes the last-compiled state

    serial_path = OUT_DIR / "f1_serial.xml"
    spec.to_file(str(serial_path))
    txt = serial_path.read_text()
    # baked meshdir: absolute at compile time -> portable "meshes/" (flat layout,
    # resolved relative to each location's own top-level xml)
    txt = txt.replace(str(MJCF_MESHES.resolve()) + "/", "meshes/")
    txt = txt.replace(str(MJCF_MESHES.resolve()), "meshes")
    serial_path.write_text(txt)
    print(f"wrote {serial_path}")

    # verify: reload via the scene wrapper (meshdir resolves from mjcf/)
    m = mujoco.MjModel.from_xml_path(str(F1 / "mjcf/f1_flat.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    foot_bodies = {body_id(m, "left_ankle_roll_link"), body_id(m, "right_ankle_roll_link")}
    soles = [d.geom_xpos[g][2] - 0.002 for g in range(m.ngeom)
             if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_bodyid[g] in foot_bodies
             and m.geom_group[g] == 3 and m.geom_contype[g] != 0]
    sole = min(soles)
    print(f"reload ok: nbody={m.nbody} njnt={m.njnt} nu={m.nu} nsensor={m.nsensor}; "
          f"keyframe stand sole z={sole:.4f} base z={d.qpos[2]:.4f}")

    # ---------------- Isaac URDF (assets/f1: urdf + flat meshes dir) ----------------
    isaac_urdf = rewrite_urdf("../meshes")
    tree = ET.fromstring(isaac_urdf)
    # IsaacLab URDF importer: keep tree; add standing joint values as a note file
    (OUT_DIR / "f1_isaac.urdf").write_text(ET.tostring(tree, encoding="unicode"))
    stand_json = {n: float(v) for n, v in zip(JOINT_ORDER, q_stand)}
    stand_json["_order"] = "X1 sensor order (legs->lumbar->arms)"
    import json
    (OUT_DIR / "f1_stand.json").write_text(json.dumps(
        {"joint_pos_sensor_order": stand_json,
         "joint_pos_urdf_order": {n: float(q_stand[JOINT_ORDER.index(n)]) for n in URDF_JOINT_ORDER},
         "base_z": float(qpos0[2]), "info": info}, indent=2))
    print("wrote f1_isaac.urdf + f1_stand.json")

    # ---------------- scene files ----------------
    env = F1 / "mjcf/environment/flat.xml"
    if not env.exists():
        env.write_text("""<mujoco model="f1 flat terrain">
  <asset>
    <texture name="groundplane" type="2d" builtin="checker" mark="edge" rgb1="0.25 0.3 0.35" rgb2="0.18 0.22 0.27" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 3 .125" type="plane" material="groundplane" conaffinity="7" condim="3" friction="1"/>
  </worldbody>
</mujoco>
""")
    wrapper = F1 / "mjcf/f1_flat.xml"
    wrapper.write_text('<mujoco model="f1 scene">\n\n    <include file="robot/f1/f1_serial.xml" />\n    <include file="environment/flat.xml" />\n\n</mujoco>\n')
    print(f"wrote {wrapper}")

    # ---- distribute: whole_body_tracking assets + GMR mocap (flat meshes dirs) ----
    WBT = REPO / "whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/f1"
    GMR_F1 = REPO / "GMR/assets/f1"
    (WBT / "mjcf/robot/f1").mkdir(parents=True, exist_ok=True)
    (WBT / "mjcf/environment").mkdir(parents=True, exist_ok=True)
    wbt_serial = (WBT / "mjcf/robot/f1/f1_serial.xml")
    wbt_serial.write_text(
        serial_path.read_text().replace('meshdir="meshes/"', 'meshdir="../meshes/"')
                                .replace('meshdir="meshes"', 'meshdir="../meshes"'))
    shutil.copy(F1 / "mjcf/environment/flat.xml", WBT / "mjcf/environment/flat.xml")
    (WBT / "mjcf/f1_flat.xml").write_text(
        '<mujoco model="f1 scene">\n\n    <include file="robot/f1/f1_serial.xml" />\n    <include file="environment/flat.xml" />\n\n</mujoco>\n')
    shutil.copy(OUT_DIR / "f1_isaac.urdf", WBT / "urdf/f1.urdf" if (WBT / "urdf").exists() else WBT / "f1_isaac.urdf")
    (WBT / "urdf").mkdir(exist_ok=True)
    shutil.copy(OUT_DIR / "f1_isaac.urdf", WBT / "urdf/f1.urdf")
    GMR_F1.mkdir(parents=True, exist_ok=True)
    shutil.copy(serial_path, GMR_F1 / "f1_mocap.xml")
    print("distributed to whole_body_tracking assets + GMR assets")


if __name__ == "__main__":
    main()
