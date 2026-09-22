import mujoco
import numpy as np

for name, xml, hip, ankle in [
    ("T1", "GMR/assets/booster_t1_29dof/t1_mocap.xml", "Hip_Yaw_Left", "left_foot_link"),
    ("X1", "GMR/assets/xyber_x1/xyber_x1_mocap.xml", "left_hip_pitch_link", "left_ankle_roll_link"),
]:
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    hi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hip)
    ai = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ankle)
    dist = np.linalg.norm(d.xpos[hi] - d.xpos[ai])
    dz = d.xpos[hi][2] - d.xpos[ai][2]
    print(f"{name}: {hip} -> {ankle}: |d|={dist:.4f} dz={dz:.4f} hip_z={d.xpos[hi][2]:.3f} ankle_z={d.xpos[ai][2]:.3f}")

# scaled human leg (LAFAN): 0.831
print("scaled legs: T1 0.6*0.831 =", 0.6 * 0.831, " X1 0.689*0.831 =", 0.689 * 0.831)
