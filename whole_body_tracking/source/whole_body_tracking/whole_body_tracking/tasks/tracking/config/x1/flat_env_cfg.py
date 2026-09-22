from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.x1 import X1_ACTION_SCALE, X1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class X1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = X1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = X1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "lumbar_pitch_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_hip_roll_link",
            "left_knee_pitch_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_pitch_link",
            "right_ankle_roll_link",
            "lumbar_pitch_link",
            "left_shoulder_roll_link",
            "left_elbow_pitch_link",
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_elbow_pitch_link",
            "right_wrist_roll_link",
        ]

        # Robot-specific overrides of G1 defaults in the base TrackingEnvCfg.
        # X1 COM randomization acts on the torso link (lumbar_pitch_link).
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("lumbar_pitch_link")

        # Unwanted contacts: everything except feet (ankle_roll) and hands (wrist_roll).
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_roll_link$)"
                r"(?!right_wrist_roll_link$).+$"
            ],
        )

        # End-effector termination: feet and hands.
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
        ]


@configclass
class X1FlatWoStateEstimationEnvCfg(X1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
