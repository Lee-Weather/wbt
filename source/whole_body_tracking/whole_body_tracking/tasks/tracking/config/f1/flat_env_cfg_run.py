from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.f1_gaits import F1_CFG_RUN, F1_ACTION_SCALE_RUN
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class F1FlatRunEnvCfg(TrackingEnvCfg):
    """F1 running env with 1.1× baseline Kp/Kd — stiffer tracking for faster motions."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = F1_CFG_RUN.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = F1_ACTION_SCALE_RUN
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

        # F1 naming matches X1: torso is lumbar_pitch_link, arm end effector is wrist_roll_link.
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
