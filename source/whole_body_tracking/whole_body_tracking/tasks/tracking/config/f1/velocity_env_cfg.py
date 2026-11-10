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


# ═══════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════


@configclass
class VelocityCommandsCfg(_CommandsCfg):
    """Velocity-command source using the reference-clip playback."""

    motion = VelocityCommandCfg(
        asset_name="robot",
        motion_file="${..motion_file}",        # filled by env config below
        # ── velocity range (m/s, rad/s) ─────────────
        vx_min=-1.0,
        vx_max=1.0,
        vy_min=-0.4,
        vy_max=0.4,
        yaw_min=-0.8,
        yaw_max=0.8,
        # ── reference clip metadata ────────────────
        ref_forward_speed=0.6,
        min_command_speed=0.05,
    )


# ═══════════════════════════════════════════════════════════════════
#  Observations
# ═══════════════════════════════════════════════════════════════════


@configclass
class VelocityObsCfg:
    """Observation groups for velocity-conditioned locomotion.

    Compared to the tracking setup, we drop the reference body-pose terms and
    keep only: velocity command, IMU-like state, joint state, and previous
    action.
    """

    @configclass
    class PolicyCfg(ObsGroup):

        # ── 3-dim velocity command ───────────────────
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
        )

        # ── proprioception ──────────────────────────
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

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

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ═══════════════════════════════════════════════════════════════════
#  Rewards
# ═══════════════════════════════════════════════════════════════════


@configclass
class VelocityRewardsCfg:
    """Reward terms: primary velocity tracking + gait prior + regularisation."""

    # ── Primary objective: velocity tracking ─────────────────
    lin_vel_tracking = RewTerm(
        func=vreward.tracking_lin_vel,
        weight=1.5,
        params={"std": 0.25},
    )
    ang_vel_tracking = RewTerm(
        func=vreward.tracking_ang_vel,
        weight=1.0,
        params={"std": 0.25},
    )

    # ── Gait prior (soft constraint from reference clip) ────
    ref_joint_prior = RewTerm(
        func=vreward.ref_joint_pos_prior,
        weight=0.2,
        params={"std": 1.0},
    )
    ref_base_height = RewTerm(
        func=vreward.ref_base_height_prior,
        weight=0.2,
        params={"std": 0.25},
    )

    # ── Regularisation ─────────────────────────────────────
    action_rate_l2 = RewTerm(
        func=vreward.action_rate,
        weight=-0.01,
    )

    # ── Feet air-time (encourages alternating swing/stance) ─
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

    # ── Termination penalty ────────────────────────────────
    undesired_contacts = RewTerm(
        func="undesired_contacts",
        weight=-0.05,
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


# ═══════════════════════════════════════════════════════════════════
#  Environment config
# ═══════════════════════════════════════════════════════════════════


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

    # ── Override components ─────────────────────────────────
    commands: VelocityCommandsCfg = VelocityCommandsCfg()
    observations: VelocityObsCfg = VelocityObsCfg()
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()

    # ── Keep existing scene / events / terminations ─────────
    # (inherited from TrackingEnvCfg)

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

        # Update termination thresholds for velocity control
        # (keep a bit looser than the tracking setup)
        self.terminations.anchor_pos.params["threshold"] = 0.30
        self.terminations.anchor_ori.params["threshold"] = 0.80

        # ── Episode length ─────────────────────────────────
        self.episode_length_s = 10.0
