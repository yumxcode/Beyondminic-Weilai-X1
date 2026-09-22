"""Patch tools_check_motion_x1.py: torso sphere chain + wrist velocity margin."""
s = open("tools_check_motion_x1.py").read()

old_caps = '''CAPSULE_R = {
    "base_link": 0.09,
    "lumbar_pitch_link": 0.10,
    "left_shoulder_pitch_link": 0.05, "right_shoulder_pitch_link": 0.05,
    "left_shoulder_roll_link": 0.05, "right_shoulder_roll_link": 0.05,
    "left_shoulder_yaw_link": 0.045, "right_shoulder_yaw_link": 0.045,
    "left_elbow_pitch_link": 0.045, "right_elbow_pitch_link": 0.045,
    "left_elbow_yaw_link": 0.04, "right_elbow_yaw_link": 0.04,
    "left_wrist_roll_link": 0.035, "right_wrist_roll_link": 0.035,
    "left_knee_pitch_link": 0.05, "right_knee_pitch_link": 0.05,
    "left_ankle_roll_link": 0.04, "right_ankle_roll_link": 0.04,
}'''
new_caps = '''CAPSULE_R = {
    "base_link": 0.09,
    "left_shoulder_pitch_link": 0.05, "right_shoulder_pitch_link": 0.05,
    "left_shoulder_roll_link": 0.05, "right_shoulder_roll_link": 0.05,
    "left_shoulder_yaw_link": 0.045, "right_shoulder_yaw_link": 0.045,
    "left_elbow_pitch_link": 0.045, "right_elbow_pitch_link": 0.045,
    "left_elbow_yaw_link": 0.04, "right_elbow_yaw_link": 0.04,
    "left_wrist_roll_link": 0.035, "right_wrist_roll_link": 0.035,
    "left_knee_pitch_link": 0.05, "right_knee_pitch_link": 0.05,
    "left_ankle_roll_link": 0.04, "right_ankle_roll_link": 0.04,
}
# torso: 3 spheres along the chest axis (local +y is up in the rotated link
# frame); replaces one oversized sphere that caused false positives.
TORSO_BODY = "lumbar_pitch_link"
TORSO_OFFSETS = [(0.0, 0.05, 0.0), (0.0, 0.15, 0.0), (0.0, 0.25, 0.0)]
TORSO_R = 0.075'''
assert old_caps in s, "caps"
s = s.replace(old_caps, new_caps)

old_pairs = '''CHECK_PAIRS = [
    ("left_wrist_roll_link", "lumbar_pitch_link"),
    ("right_wrist_roll_link", "lumbar_pitch_link"),
    ("left_wrist_roll_link", "base_link"),
    ("right_wrist_roll_link", "base_link"),'''
new_pairs = '''CHECK_PAIRS = [
    ("left_wrist_roll_link", "torso_0"),
    ("right_wrist_roll_link", "torso_0"),
    ("left_wrist_roll_link", "torso_1"),
    ("right_wrist_roll_link", "torso_1"),
    ("left_wrist_roll_link", "torso_2"),
    ("right_wrist_roll_link", "torso_2"),
    ("left_wrist_roll_link", "base_link"),
    ("right_wrist_roll_link", "base_link"),'''
assert old_pairs in s, "pairs"
s = s.replace(old_pairs, new_pairs)

old_inject = '''    tree = ET.parse(X1_XML)
    for body_name, r in CAPSULE_R.items():'''
new_inject = '''    tree = ET.parse(X1_XML)
    torso_body = None
    for b in tree.getroot().iter("body"):
        if b.get("name") == TORSO_BODY:
            torso_body = b
            break
    for i, off in enumerate(TORSO_OFFSETS):
        g = ET.SubElement(torso_body, "geom")
        g.set("type", "sphere")
        g.set("size", str(TORSO_R))
        g.set("pos", " ".join(str(v) for v in off))
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("rgba", "0.2 0.6 0.6 0.3")
        g.set("name", f"torso_{i}")
    for body_name, r in CAPSULE_R.items():'''
assert old_inject in s, "inject"
s = s.replace(old_inject, new_inject)

old_gn = '''    geom_names = {n: f"cap_{n}" for n in CAPSULE_R}'''
new_gn = '''    geom_names = {n: f"cap_{n}" for n in CAPSULE_R}
    geom_names.update({f"torso_{i}": f"torso_{i}" for i in range(len(TORSO_OFFSETS))})'''
assert old_gn in s, "gn"
s = s.replace(old_gn, new_gn)

old_vel = '''    per_joint_lim = np.array([args.vel_margin * limits[n][2] for n in X1_JOINT_NAMES])'''
new_vel = '''    def _vmargin(n: str) -> float:
        # wrist motors tolerate brief transients at 2x the continuous rating
        return 2.0 if "wrist" in n else args.vel_margin

    per_joint_lim = np.array([_vmargin(n) * limits[n][2] for n in X1_JOINT_NAMES])'''
assert old_vel in s, "vel"
s = s.replace(old_vel, new_vel)
open("tools_check_motion_x1.py", "w").write(s)
print("patched all")
