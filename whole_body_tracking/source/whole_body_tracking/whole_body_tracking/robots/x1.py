import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Armature values chosen so that the resulting PD gains (natural frequency
# 10 Hz, damping ratio 2.0) match the official AgiBot X1 stand controller:
#   hip_pitch 30, hip_roll 40, hip_yaw 35, knee_pitch 100, ankle 35 [N*m/rad]
ARM_HIP_PITCH = 0.00760
ARM_HIP_ROLL = 0.01013
ARM_HIP_YAW = 0.00887
ARM_KNEE = 0.02533
ARM_ANKLE = 0.00887
ARM_LUMBAR_YAW_ROLL = 0.01013
ARM_LUMBAR_PITCH = 0.01266
ARM_ARM = 0.00253
ARM_WRIST = 0.00127

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


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
                ".*_hip_pitch_joint": _stiffness(ARM_HIP_PITCH),
                ".*_hip_roll_joint": _stiffness(ARM_HIP_ROLL),
                ".*_hip_yaw_joint": _stiffness(ARM_HIP_YAW),
                ".*_knee_pitch_joint": _stiffness(ARM_KNEE),
            },
            damping={
                ".*_hip_pitch_joint": _damping(ARM_HIP_PITCH),
                ".*_hip_roll_joint": _damping(ARM_HIP_ROLL),
                ".*_hip_yaw_joint": _damping(ARM_HIP_YAW),
                ".*_knee_pitch_joint": _damping(ARM_KNEE),
            },
            armature={
                ".*_hip_pitch_joint": ARM_HIP_PITCH,
                ".*_hip_roll_joint": ARM_HIP_ROLL,
                ".*_hip_yaw_joint": ARM_HIP_YAW,
                ".*_knee_pitch_joint": ARM_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=80.0,
            velocity_limit_sim=13.6,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=_stiffness(ARM_ANKLE),
            damping=_damping(ARM_ANKLE),
            armature=ARM_ANKLE,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=150.0,
            velocity_limit_sim=8.9,
            joint_names_expr=["lumbar_yaw_joint", "lumbar_roll_joint"],
            stiffness=_stiffness(ARM_LUMBAR_YAW_ROLL),
            damping=_damping(ARM_LUMBAR_YAW_ROLL),
            armature=ARM_LUMBAR_YAW_ROLL,
        ),
        "waist_pitch": ImplicitActuatorCfg(
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            joint_names_expr=["lumbar_pitch_joint"],
            stiffness=_stiffness(ARM_LUMBAR_PITCH),
            damping=_damping(ARM_LUMBAR_PITCH),
            armature=ARM_LUMBAR_PITCH,
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
                ".*_shoulder_pitch_joint": _stiffness(ARM_ARM),
                ".*_shoulder_roll_joint": _stiffness(ARM_ARM),
                ".*_shoulder_yaw_joint": _stiffness(ARM_ARM),
                ".*_elbow_pitch_joint": _stiffness(ARM_ARM),
                ".*_elbow_yaw_joint": _stiffness(ARM_ARM),
                ".*_wrist_pitch_joint": _stiffness(ARM_WRIST),
                ".*_wrist_roll_joint": _stiffness(ARM_WRIST),
            },
            damping={
                ".*_shoulder_pitch_joint": _damping(ARM_ARM),
                ".*_shoulder_roll_joint": _damping(ARM_ARM),
                ".*_shoulder_yaw_joint": _damping(ARM_ARM),
                ".*_elbow_pitch_joint": _damping(ARM_ARM),
                ".*_elbow_yaw_joint": _damping(ARM_ARM),
                ".*_wrist_pitch_joint": _damping(ARM_WRIST),
                ".*_wrist_roll_joint": _damping(ARM_WRIST),
            },
            armature={
                ".*_shoulder_pitch_joint": ARM_ARM,
                ".*_shoulder_roll_joint": ARM_ARM,
                ".*_shoulder_yaw_joint": ARM_ARM,
                ".*_elbow_pitch_joint": ARM_ARM,
                ".*_elbow_yaw_joint": ARM_ARM,
                ".*_wrist_pitch_joint": ARM_WRIST,
                ".*_wrist_roll_joint": ARM_WRIST,
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
