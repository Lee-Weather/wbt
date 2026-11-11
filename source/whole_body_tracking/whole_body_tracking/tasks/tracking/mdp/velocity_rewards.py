"""
Reward functions for velocity-conditioned locomotion training.

Combines:
1. Velocity tracking reward (primary objective)
2. Shaped auxiliary rewards for stable gait (orientation, smoothness, efficiency)
3. Survival bonus to bootstrap learning
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from whole_body_tracking.tasks.tracking.mdp.velocity_commands import VelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ═══════════════════════════════════════════════════════════════════
#  Primary: velocity tracking
# ═══════════════════════════════════════════════════════════════════

def tracking_lin_vel(env: ManagerBasedRLEnv, command_name: str,
                     std: float = 0.25) -> torch.Tensor:
    """Exponential reward for matching commanded linear velocity.

    Uses the robot's base-frame forward/leftward velocity (root_lin_vel_b)
    and compares it to the commanded (vx, vy) from VelocityCommand.

    Args:
        std:  the smaller the std, the sharper the reward peak.
    """
    cmd: VelocityCommand = env.command_manager.get_term(command_name)
    robot_xy = cmd.robot.data.root_lin_vel_b[:, :2]        # (N, 2)
    target = torch.stack([cmd.cmd_vx, cmd.cmd_vy], dim=1)   # (N, 2)
    error = torch.sum((robot_xy - target) ** 2, dim=-1)
    return torch.exp(-error / std ** 2)


def tracking_ang_vel(env: ManagerBasedRLEnv, command_name: str,
                     std: float = 0.25) -> torch.Tensor:
    """Exponential reward for matching commanded yaw rate."""
    cmd: VelocityCommand = env.command_manager.get_term(command_name)
    robot_yaw = cmd.robot.data.root_ang_vel_b[:, 2]
    error = (robot_yaw - cmd.cmd_yaw) ** 2
    return torch.exp(-error / std ** 2)


# ═══════════════════════════════════════════════════════════════════
#  Reference prior (soft gait regularisation)
# ═══════════════════════════════════════════════════════════════════

def ref_joint_pos_prior(env: ManagerBasedRLEnv, command_name: str,
                        std: float = 1.0) -> torch.Tensor:
    """Exponential penalty for deviating from the reference joint positions.

    This is the key "gait prior": it keeps the policy near the natural walking
    kinematics of the reference clip while the velocity reward drives it to
    produce the *commanded* speed.
    """
    cmd: VelocityCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        (cmd.ref_joint_pos - cmd.robot.data.joint_pos) ** 2, dim=-1
    )
    return torch.exp(-error / std ** 2)


def ref_base_height_prior(env: ManagerBasedRLEnv, command_name: str,
                          std: float = 0.25) -> torch.Tensor:
    """Exponential penalty for deviating from the reference base height."""
    cmd: VelocityCommand = env.command_manager.get_term(command_name)
    error = (cmd.ref_anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2]) ** 2
    return torch.exp(-error / std ** 2)


# ═══════════════════════════════════════════════════════════════════
#  Regularisation
# ═══════════════════════════════════════════════════════════════════

def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise large changes in action (smoothness)."""
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2,
        dim=-1,
    )


def feet_air_time(env: ManagerBasedRLEnv,
                  sensor_cfg: SceneEntityCfg,
                  threshold: float = 0.3) -> torch.Tensor:
    """Reward the robot for keeping feet off the ground (swing phase).

    This encourages a healthy alternating gait rather than dragging feet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(
        env.step_dt / env.physics_dt, env.physics_dt
    )[:, sensor_cfg.body_ids]
    last_contact = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    air_time = torch.where(
        last_contact > threshold,
        first_air.float(),
        first_air.float() - last_contact + threshold,
    )
    return torch.sum(air_time, dim=-1)


# ═══════════════════════════════════════════════════════════════════
#  Survival / regularisation
# ═══════════════════════════════════════════════════════════════════


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Positive reward for staying alive (not terminated).

    This is the primary gradient that drives exploration.  Without it,
    the robot gets ~zero tracking reward early in training (when it can't
    yet walk) and has no incentive to keep trying.
    """
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device) * 2.0


# ═══════════════════════════════════════════════════════════════════
#  Shaping rewards for gait quality
# ═══════════════════════════════════════════════════════════════════


def dof_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise large joint torques (energy efficiency)."""
    return torch.sum(env.scene["robot"].data.applied_torque ** 2, dim=-1)


def lin_vel_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise vertical velocity (discourage hopping / bouncing)."""
    return env.scene["robot"].data.root_lin_vel_b[:, 2] ** 2


def dof_vel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise large joint velocities."""
    return torch.sum(env.scene["robot"].data.joint_vel ** 2, dim=-1)


def orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the robot for keeping its torso upright.

    Computes the dot product of the current base z-axis with the world
    gravity direction (0,0,1).  When perfectly upright returns 1,
    when upside down returns -1.
    """
    # Base z-axis in world frame
    quat = env.scene["robot"].data.root_quat_w

    # Manual quat_apply: rotate local z (0,0,1) by quat
    z_local = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).expand(env.num_envs, -1)

    w = quat[:, 0:1]
    x = quat[:, 1:2]
    y = quat[:, 2:3]
    z = quat[:, 3:4]

    # t = 2 * cross(xyz, v)
    tx = 2 * (y * z_local[:, 2:3] - z * z_local[:, 1:2])
    ty = 2 * (z * z_local[:, 0:1] - x * z_local[:, 2:3])
    tz = 2 * (x * z_local[:, 1:2] - y * z_local[:, 0:1])

    # v' = v + w * t + cross(xyz, t)
    vz = z_local[:, 2:3] + w * tz + x * ty - y * tx

    return vz.squeeze(-1)
