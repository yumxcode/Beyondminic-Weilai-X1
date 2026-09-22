"""Distribution of per-frame min self-pair distances for dance1_subject3_x1."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "GMR")
import mujoco
import xml.etree.ElementTree as ET
import tempfile

REPO = Path(__file__).resolve().parent
exec(open(REPO / "tools_check_motion_x1.py").read().split("def urdf_limits")[0])  # reuse constants

# Rebuild the checker model (spheres) quickly
tree = ET.parse(X1_XML)
torso_body = None
for b in tree.getroot().iter("body"):
    if b.get("name") == TORSO_BODY:
        torso_body = b
for i, off in enumerate(TORSO_OFFSETS):
    g = ET.SubElement(torso_body, "geom")
    g.set("type", "sphere"); g.set("size", str(TORSO_R))
    g.set("pos", " ".join(str(v) for v in off))
    g.set("contype", "0"); g.set("conaffinity", "0")
    g.set("name", f"torso_{i}")
for body_name, r in CAPSULE_R.items():
    for b in tree.getroot().iter("body"):
        if b.get("name") == body_name:
            g = ET.SubElement(b, "geom")
            g.set("type", "sphere"); g.set("size", str(r))
            g.set("contype", "0"); g.set("conaffinity", "0")
            g.set("name", f"cap_{body_name}")
            break
tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=str(X1_XML.parent))
tree.write(tmp.name); tmp.close()
model = mujoco.MjModel.from_xml_path(tmp.name)
Path(tmp.name).unlink()
data = mujoco.MjData(model)

ORDER = ["lumbar_yaw_joint","lumbar_roll_joint","lumbar_pitch_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_pitch_joint","left_elbow_yaw_joint","left_wrist_pitch_joint","left_wrist_roll_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_pitch_joint","right_elbow_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint","left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_pitch_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_pitch_joint","right_ankle_pitch_joint","right_ankle_roll_joint"]
qadr = {}
for jid in range(model.njnt):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if n and n != "floating_base":
        qadr[n] = int(model.jnt_qposadr[jid])

csv = np.loadtxt(REPO / "whole_body_tracking/motions/csv/dance1_subject3_x1.csv", delimiter=",")
mins = []
worst_pairs = []
for f in range(0, len(csv), 3):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = csv[f, 0:3]
    q = csv[f, 3:7]
    data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
    for i, n in enumerate(ORDER):
        data.qpos[qadr[n]] = csv[f, 7 + i]
    mujoco.mj_forward(model, data)
    frame_min, frame_pair = np.inf, None
    for a, b in CHECK_PAIRS:
        ga = a if a.startswith("torso_") else f"cap_{a}"
        gb = b if b.startswith("torso_") else f"cap_{b}"
        g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ga)
        g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gb)
        assert g1 >= 0 and g2 >= 0, (ga, gb)
        dist = mujoco.mj_geomDistance(model, data, g1, g2, 5.0, None)
        if dist < frame_min:
            frame_min, frame_pair = dist, (a, b)
    mins.append((f, frame_min, frame_pair))

vals = np.array([m[1] for m in mins])
print(f"min-self distribution over {len(vals)} frames:")
for pct in (0, 1, 5, 10, 25, 50):
    print(f"  p{pct:02d}: {np.percentile(vals, pct):.4f} m")
neg = [m for m in mins if m[1] < 0]
print(f"negative frames: {len(neg)}")
if neg:
    deep = sorted(neg, key=lambda x: x[1])[:10]
    print("deepest 10 (frame, dist, pair):")
    for f, d, p in deep:
        print(f"  {f:5d} {d:8.4f} {p}")
    from collections import Counter
    print("pair histogram:", Counter(p for _, _, p in neg).most_common(5))
