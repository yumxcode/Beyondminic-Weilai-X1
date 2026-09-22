#!/usr/bin/env python3
"""Build the GMR retargeting MJCF for the Xyber X1 (AgiBot X1) 29-DoF robot.

Takes X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml and produces
GMR/assets/xyber_x1/xyber_x1_mocap.xml with:
  - meshdir pointing to a local meshes/ copy (only referenced robot meshes)
  - joint ranges taken from the (real-robot) URDF limits instead of the looser MJCF ones
  - actuators (motors) with URDF effort limits so GMR's robot_motor_names is populated
"""
from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC_XML = REPO / "X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml"
SRC_MESHES = REPO / "X1_29DOF/meshes"
URDF = REPO / "X1_29DOF/urdf/f1.urdf"
DST_DIR = REPO / "GMR/assets/xyber_x1"
DST_XML = DST_DIR / "xyber_x1_mocap.xml"
DST_MESHES = DST_DIR / "meshes"


def urdf_limits() -> dict[str, tuple[float, float, float]]:
    """joint name -> (lower, upper, effort)."""
    tree = ET.parse(URDF)
    out = {}
    for j in tree.getroot().iter("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        lim = j.find("limit")
        out[j.get("name")] = (
            float(lim.get("lower")),
            float(lim.get("upper")),
            float(lim.get("effort")),
        )
    return out


def main() -> None:
    limits = urdf_limits()
    tree = ET.parse(SRC_XML)
    root = tree.getroot()
    root.set("model", "xyber_x1")

    compiler = root.find("compiler")
    compiler.set("meshdir", "meshes")

    # collect referenced meshes and fix case-sensitivity (some URDFs use .stl vs .STL)
    referenced = []
    for mesh in root.iter("mesh"):
        file = mesh.get("file")
        referenced.append(file)
    DST_MESHES.mkdir(parents=True, exist_ok=True)
    lower_map = {p.name.lower(): p.name for p in SRC_MESHES.iterdir() if p.is_file()}
    for rel in sorted(set(referenced)):
        src = SRC_MESHES / lower_map[rel.lower()]
        shutil.copy2(src, DST_MESHES / src.name)
        mesh_els = [m for m in root.iter("mesh") if m.get("file") == rel]
        for m in mesh_els:
            m.set("file", src.name)

    # apply URDF joint ranges (skip the floating base)
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name in limits:
            lo, up, _ = limits[name]
            # small safety margin so mink IK iterations never trip on limits;
            # the CSV writer clamps final values back to the real URDF limits.
            margin = 0.01
            joint.set("range", f"{lo - margin} {up + margin}")
        elif joint.get("type") == "hinge" and "range" not in joint.attrib:
            print(f"[warn] hinge joint without URDF limit: {name}")

    # rebuild actuators from URDF effort limits, keeping MJCF joint tree order
    worldbody = root.find("worldbody")
    joint_order = [j.get("name") for j in worldbody.iter("joint") if j.get("name")]
    old_act = root.find("actuator")
    if old_act is not None:
        root.remove(old_act)
    act = ET.SubElement(root, "actuator")
    for name in joint_order:
        if name in limits:
            effort = limits[name][2]
            motor = ET.SubElement(act, "motor")
            motor.set("name", f"motor_{name}")
            motor.set("joint", name)
            motor.set("ctrlrange", f"{-effort} {effort}")
            motor.set("ctrllimited", "true")

    # drop sensors/keyframe sections: not needed for retargeting (keep model minimal)
    for tag in ("sensor", "keyframe"):
        el = root.find(tag)
        if el is not None:
            root.remove(el)

    ET.indent(tree, space="  ")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    tree.write(DST_XML, encoding="utf-8", xml_declaration=True)
    print(f"wrote {DST_XML}")
    print(f"meshes copied: {len(set(referenced))}")
    print(f"joints with URDF limits: {sum(1 for n in joint_order if n in limits)}")


if __name__ == "__main__":
    main()
