import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Identified actuator parameters (2026-10-14, x1_armature_pd.md "方案 B"):
# lab 12-DOF torque identification + X1 walking-log estimates after sim2real.
# kd capped at 8.0 (MIT encoding limit on R86/R52 hosts, motor range fixed);
# kp lowered on joints whose kd demand exceeded the cap so the damping ratio
# stays 0.4-0.7 under the identified inertia. Effort / velocity limits are
# unchanged from the previous config.
# Keep this table in sync with scripts/play_mujoco.py (_X1_JOINT_CLASS).


X1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/xyber_x1/urdf/xyber_x1.urdf",
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
        pos=(0.0, 0.0, 0.61),
        joint_pos={
            "left_hip_pitch_joint": 0.4889,
            "left_hip_roll_joint": 0.0621,
            "left_hip_yaw_joint": -0.3385,
            "left_knee_pitch_joint": 0.632,
            "left_ankle_pitch_joint": -0.2722,
            "right_hip_pitch_joint": -0.4889,
            "right_hip_roll_joint": -0.0621,
            "right_hip_yaw_joint": 0.3385,
            "right_knee_pitch_joint": 0.632,
            "right_ankle_pitch_joint": -0.2722,
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
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 180.0,
                ".*_knee_pitch_joint": 180.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 27.2,
                ".*_hip_roll_joint": 27.2,
                ".*_hip_pitch_joint": 8.9,
                ".*_knee_pitch_joint": 8.9,
            },
            stiffness={
                ".*_hip_pitch_joint": 65.0,   # 120 -> 65 (kd-capped, ζ≈0.71)
                ".*_hip_roll_joint": 60.0,    # 100 -> 60 (kd-capped, ζ≈0.60)
                ".*_hip_yaw_joint": 100.0,    # unchanged (ζ≈0.81)
                ".*_knee_pitch_joint": 85.0,  # 150 -> 85 (kd-capped, ζ≈0.70)
            },
            damping={
                ".*_hip_pitch_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_yaw_joint": 3.3,
                ".*_knee_pitch_joint": 8.0,
            },
            armature={
                ".*_hip_pitch_joint": 0.208,   # lab identification
                ".*_hip_roll_joint": 0.25,     # log estimate
                ".*_hip_yaw_joint": 0.0148,    # lab identification
                ".*_knee_pitch_joint": 0.273,  # lab identification
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=80.0,
            velocity_limit_sim=13.6,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=50.0,
            damping={
                ".*_ankle_pitch_joint": 4.0,  # 2.5-equivalent -> 4.0 (ζ≈0.73)
                ".*_ankle_roll_joint": 2.5,   # unchanged (ζ≈0.93)
            },
            armature={
                ".*_ankle_pitch_joint": 0.15,  # lab identification
                ".*_ankle_roll_joint": 0.035,  # lab identification
            },
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=150.0,
            velocity_limit_sim=8.9,
            joint_names_expr=["lumbar_yaw_joint", "lumbar_roll_joint"],
            stiffness={
                "lumbar_yaw_joint": 90.0,   # 120 -> 90 (ζ≈0.49)
                "lumbar_roll_joint": 70.0,  # 120 -> 70 (ζ≈0.41)
            },
            damping=8.0,
            armature={
                "lumbar_yaw_joint": 0.022,  # log estimate
                "lumbar_roll_joint": 0.021,  # log estimate
            },
        ),
        "waist_pitch": ImplicitActuatorCfg(
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            joint_names_expr=["lumbar_pitch_joint"],
            stiffness=100.0,  # 150 -> 100 (ζ≈0.46)
            damping=8.0,
            armature=0.012,   # log estimate
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
                ".*_shoulder_pitch_joint": 4.0,   # 2.0-equivalent -> 4.0 (ζ≈0.72)
                ".*_shoulder_roll_joint": 4.0,    # 2.0-equivalent -> 4.0 (ζ≈0.71)
                ".*_shoulder_yaw_joint": 2.0,     # unchanged (ζ≈1.08)
                ".*_elbow_pitch_joint": 1.9,      # 1.5-equivalent -> 1.9 (ζ≈0.69)
                ".*_elbow_yaw_joint": 1.0,        # unchanged (ζ≈1.06)
                ".*_wrist_pitch_joint": 1.0,
                ".*_wrist_roll_joint": 1.0,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.02,   # right-arm log estimate, mirrored
                ".*_shoulder_roll_joint": 0.013,   # right-arm log estimate, mirrored
                ".*_shoulder_yaw_joint": 0.01,     # below resolution, kept
                ".*_elbow_pitch_joint": 0.01,      # below resolution, kept
                ".*_elbow_yaw_joint": 0.01,        # below resolution, kept
                ".*_wrist_pitch_joint": 0.01,      # kept; L28 runs internal position loop
                ".*_wrist_roll_joint": 0.01,
            },
        ),
    },
)

X1_ACTION_SCALE = {}
for a in X1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            X1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
