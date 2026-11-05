import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# F1 leg gains (2026-11-05, F1_29DOF/F1_AUDIT.md + f1_kp_kd.json):
# URDF standing-pose subtree inertia + estimated rotor armature (scaled from
# X1 identified values by torque class 130/180 — no lab ID for F1), kp in the
# X1-proven bandwidth band (hips ~1.3 Hz, knee 2.3 Hz, ankles ~10 Hz on ~10x
# lighter ankles), kd from zeta 0.4-0.55 capped at 8 (MIT encoding, same
# host-stack constraint as X1). Upper body: links/joints byte-identical to X1
# -> X1 PlanB identified gains reused verbatim.
# Keep this table in sync with scripts/play_mujoco.py (_F1_JOINT_CLASS).

F1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/f1/urdf/f1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.588),  # standing keyframe (F1_29DOF/mjcf/robot/f1/f1_stand.json)
        joint_pos={
            "left_hip_pitch_joint": -0.2915,
            "left_knee_pitch_joint": 0.4500,
            "left_ankle_pitch_joint": -0.1585,
            "right_hip_pitch_joint": -0.2915,
            "right_knee_pitch_joint": 0.4500,
            "right_ankle_pitch_joint": -0.1585,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 130.0,
                ".*_hip_roll_joint": 130.0,
                ".*_hip_pitch_joint": 130.0,
                ".*_knee_pitch_joint": 130.0,
            },
            velocity_limit_sim=14.66,
            stiffness={
                ".*_hip_pitch_joint": 70.0,
                ".*_hip_roll_joint": 60.0,
                ".*_hip_yaw_joint": 90.0,
                ".*_knee_pitch_joint": 90.0,
            },
            damping={
                ".*_hip_pitch_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_yaw_joint": 2.0,
                ".*_knee_pitch_joint": 6.0,
            },
            armature={
                ".*_hip_pitch_joint": 0.15,   # scaled from X1 ID (0.208 * 130/180)
                ".*_hip_roll_joint": 0.15,    # estimate
                ".*_hip_yaw_joint": 0.02,     # estimate
                ".*_knee_pitch_joint": 0.20,  # scaled from X1 ID (0.273 * 130/180)
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim={
                ".*_ankle_pitch_joint": 241.875,
                ".*_ankle_roll_joint": 218.4,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 7.57,
                ".*_ankle_roll_joint": 8.05,
            },
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 45.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.0,   # F1 ankle I_eff ~0.014 -> zeta 0.55
                ".*_ankle_roll_joint": 0.6,    # I_eff ~0.011 -> zeta 0.42
            },
            armature={
                ".*_ankle_pitch_joint": 0.01,  # estimate (X1 ID 0.15 incl. heavy foot)
                ".*_ankle_roll_joint": 0.01,   # estimate
            },
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=150.0,
            velocity_limit_sim=8.9,
            joint_names_expr=["lumbar_yaw_joint", "lumbar_roll_joint"],
            stiffness={
                "lumbar_yaw_joint": 90.0,
                "lumbar_roll_joint": 70.0,
            },
            damping=8.0,
            armature={
                "lumbar_yaw_joint": 0.022,
                "lumbar_roll_joint": 0.021,
            },
        ),
        "waist_pitch": ImplicitActuatorCfg(
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            joint_names_expr=["lumbar_pitch_joint"],
            stiffness=100.0,
            damping=8.0,
            armature=0.012,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 20.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_pitch_joint": 20.0,
                ".*_elbow_yaw_joint": 20.0,
                ".*_wrist_pitch_joint": 10.0,
                ".*_wrist_roll_joint": 10.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.6,
                ".*_shoulder_roll_joint": 13.6,
                ".*_shoulder_yaw_joint": 13.6,
                ".*_elbow_pitch_joint": 13.6,
                ".*_elbow_yaw_joint": 13.6,
                ".*_wrist_pitch_joint": 2.5,
                ".*_wrist_roll_joint": 2.5,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
                ".*_wrist_pitch_joint": 15.0,
                ".*_wrist_roll_joint": 15.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 4.0,
                ".*_shoulder_roll_joint": 4.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_pitch_joint": 1.9,
                ".*_elbow_yaw_joint": 1.0,
                ".*_wrist_pitch_joint": 1.0,
                ".*_wrist_roll_joint": 1.0,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.02,
                ".*_shoulder_roll_joint": 0.013,
                ".*_shoulder_yaw_joint": 0.01,
                ".*_elbow_pitch_joint": 0.01,
                ".*_elbow_yaw_joint": 0.01,
                ".*_wrist_pitch_joint": 0.01,
                ".*_wrist_roll_joint": 0.01,
            },
        ),
    },
)

F1_ACTION_SCALE = {}
for a in F1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            F1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
