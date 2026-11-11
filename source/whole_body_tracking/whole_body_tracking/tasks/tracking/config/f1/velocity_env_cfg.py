"""
Velocity-conditioned locomotion environment for F1 (29-DoF) humanoid.

Trains a policy that takes (vx, vy, ω_yaw) velocity commands as input and
outputs joint-position targets that achieve the commanded speed while
maintaining a natural gait through auxiliary reference-trajectory priors.
"""

from isaaclab.managers import SceneEntityCfg as SceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.f1 import F1_ACTION_SCALE, F1_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    CommandsCfg as _CommandsCfg,     # base class (we override motion)
    ActionsCfg,
    EventCfg,
    TerminationsCfg,
    MySceneCfg,
    TrackingEnvCfg,
)
from whole_body_tracking.tasks.tracking.mdp.velocity_commands import (
    VelocityCommandCfg,
)
from whole_body_tracking.tasks.tracking.mdp import velocity_rewards as vreward


# Commands


@configclass
class VelocityCommandsCfg(_CommandsCfg):
    """Velocity-command source using the reference-clip playback."""

    motion = VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file="${..motion_file}",        # filled by env config below
        # velocity range (m/s, rad/s)
        vx_min=-0.5,
        vx_max=0.5,
        vy_min=-0.2,
        vy_max=0.2,
        yaw_min=-0.4,
        yaw_max=0.4,
        # reference clip metadata
        ref_forward_speed=0.6,
        min_command_speed=0.05,
    )


# Observations


@configclass
class VelocityObsCfg:
    """Observation groups for velocity-conditioned locomotion.

    Compared to the tracking setup, we drop the reference body-pose terms and
    keep only: velocity command, IMU-like state, joint state, and previous
    action.
    """

    @configclass
    class PolicyCfg(ObsGroup):

        # 3-dim velocity command
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
        )

        # proprioception (reduced noise)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=mdp.last_action)
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={"command_name": "motion"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic sees the same (non-noisy) view as the policy."""
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={"command_name": "motion"},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# Rewards


@configclass
class VelocityRewardsCfg:
    """Reward terms: alive bonus + velocity tracking + shaping + regularisation."""

    # Survival incentive
    alive = RewTerm(
        func=vreward.alive_bonus,
        weight=1.0,
    )

    # Primary objective: velocity tracking
    lin_vel_tracking = RewTerm(
        func=vreward.tracking_lin_vel,
        weight=1.0,
        params={"command_name": "motion", "std": 0.25},
    )
    ang_vel_tracking = RewTerm(
        func=vreward.tracking_ang_vel,
        weight=1.5,
        params={"command_name": "motion", "std": 0.25},
    )

    # Smoothness / efficiency
    action_rate_l2 = RewTerm(
        func=vreward.action_rate,
        weight=-0.005,
    )
    dof_torques_l2 = RewTerm(
        func=vreward.dof_torques_l2,
        weight=-2.5e-5,
    )
    dof_vel_l2 = RewTerm(
        func=vreward.dof_vel_l2,
        weight=-1e-4,
    )

    # Gait quality
    feet_air_time = RewTerm(
        func=vreward.feet_air_time,
        weight=0.05,
        params={
            "sensor_cfg": SceneCfg(
                "contact_forces",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            ),
            "threshold": 0.3,
        },
    )

    # Stability shaping
    lin_vel_z = RewTerm(
        func=vreward.lin_vel_z,
        weight=-1.0,
    )
    orientation = RewTerm(
        func=vreward.orientation,
        weight=0.5,
    )

    # Termination penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_wrist_roll_link$)(?!right_wrist_roll_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

    # Joint position limits
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneCfg("robot", joint_names=[".*"])},
    )


# Environment config


@configclass
class F1VelocityEnvCfg(TrackingEnvCfg):
    """F1 29-DoF velocity-conditioned locomotion environment.

    Usage
    ------
    .. code:: bash

        python scripts/rsl_rl/train.py \
            --task=Tracking-Flat-F1-Velocity-v0 \
            --headless
    """

    # Override components
    commands: VelocityCommandsCfg = VelocityCommandsCfg()
    observations: VelocityObsCfg = VelocityObsCfg()
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()

    # Keep existing scene / events / terminations
    # (inherited from TrackingCfg)

    def __post_init__(self):
        super().__post_init__()

        # Wire robot config
        self.scene.robot = F1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = F1_ACTION_SCALE

        # Point motion_file to a default reference clip
        from whole_body_tracking.assets import ASSET_DIR
        self.commands.motion.motion_file = (
            f"{ASSET_DIR}/f1/motions/converted/B3_walk1_weichai_f1_29.npz"
        )

        # F1 has lumbar_pitch_link instead of torso_link
        self.events.base_com.params["asset_cfg"] = SceneCfg("robot", body_names="lumbar_pitch_link")

        # Override terminations with velocity-specific functions
        # that check robot state directly (no dependency on reference-trajectory attributes).
        from whole_body_tracking.tasks.tracking.mdp import (
            base_height_fall,
            base_tilt,
            limbs_below_ground,
        )
        self.terminations.anchor_pos = DoneTerm(
            func=base_height_fall,
            params={"threshold": 0.30},
        )
        self.terminations.anchor_ori = DoneTerm(
            func=base_tilt,
            params={"threshold": 1.0},
        )
        self.terminations.ee_body_pos = DoneTerm(
            func=limbs_below_ground,
            params={"threshold": 0.10},
        )

        # Episode length
        self.episode_length_s = 10.0
