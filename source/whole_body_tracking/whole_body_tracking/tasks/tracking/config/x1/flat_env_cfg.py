from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.x1 import X1_ACTION_SCALE, X1_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class X1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = X1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = X1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
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

        # X1 naming: torso is lumbar_pitch_link, arm end effector is wrist_roll_link.
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="lumbar_pitch_link")
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_roll_link$)(?!right_wrist_roll_link$).+$"
            ],
        )
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
        ]
