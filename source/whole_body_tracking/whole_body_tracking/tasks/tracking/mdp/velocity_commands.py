"""
Velocity command term for velocity-conditioned locomotion training.

Loads a reference motion clip and provides:
1. Velocity commands (vx, vy, yaw) as policy observation (3-dim)
2. Reference joint/body state for auxiliary prior rewards

The reference playback speed is scaled by the velocity command magnitude:
  fast command → faster playback → reference steps forward quicker
  slow command → slower playback → reference creeps forward

This keeps the reference trajectory as a "soft gait prior" while the policy
learns to track the commanded velocity.
"""

from __future__ import annotations


import numpy as np
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Observation helper ─────────────────────────────────────────────


def ref_joint_pos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the reference joint positions from the velocity command.

    Provides a strong prior on the target pose so the policy can focus on
    modulating velocity rather than discovering gait from scratch.
    """
    cmd: VelocityCommand = env.command_manager.get_term(command_name)
    return cmd.ref_joint_pos


def ref_joint_vel(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the reference joint velocities from the velocity command."""
    cmd: VelocityCommand = env.command_manager.get_term(command_name)
    return cmd.ref_joint_vel


class VelocityCommand(CommandTerm):
    """Command term that provides velocity commands + scaled reference playback."""

    cfg: VelocityCommandCfg

    def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # ── Load NPZ reference motion ──────────────────────────────
        data = np.load(cfg.motion_file)

        # The clip stores joints in its own order; the articulation exposes its
        # own. Reorder by name (same rule as MotionLoader) — otherwise the
        # reference pose is permuted and every episode dies on the first step.
        npz_joint_names = data["joint_names"] if "joint_names" in data.files else None
        if npz_joint_names is not None:
            missing = [n for n in self.robot.joint_names if n not in set(npz_joint_names.tolist())]
            if missing:
                raise ValueError(f"reference clip is missing robot joints: {missing}")
            joint_order = [int(np.where(npz_joint_names == name)[0][0]) for name in self.robot.joint_names]
        else:
            joint_order = list(range(self.robot.num_joints))

        self._ref_joint_pos = torch.tensor(
            data["joint_pos"][:, joint_order].copy(), dtype=torch.float32, device=self.device
        )
        self._ref_joint_vel = torch.tensor(
            data["joint_vel"][:, joint_order].copy(), dtype=torch.float32, device=self.device
        )
        # root (first body) trajectory: NPZ stores body_pos_w[:, 0], body_quat_w[:, 0]
        self._ref_root_pos = torch.tensor(
            data["body_pos_w"][:, 0].copy(), dtype=torch.float32, device=self.device
        )
        self._ref_root_quat = torch.tensor(
            data["body_quat_w"][:, 0].copy(), dtype=torch.float32, device=self.device
        )

        self._n_frames = self._ref_joint_pos.shape[0]
        fps = float(data.get("fps", 50.0))

        # ── Precompute per-frame velocity magnitude ────────────────
        root_disp = torch.norm(
            self._ref_root_pos[1:] - self._ref_root_pos[:-1], dim=-1
        )
        dt_frame = 1.0 / fps
        self._ref_forward_vel = torch.mean(root_disp) / dt_frame  # scalar m/s

        # ── Per-env state ──────────────────────────────────────────
        self._time_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._vx = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._vy = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # metrics
        self.metrics["vx_err"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["vy_err"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["yaw_err"] = torch.zeros(self.num_envs, device=self.device)

    # ── Public properties (read by observation/reward terms) ────

    @property
    def command(self) -> torch.Tensor:
        """Policy observation: (vx, vy, yaw) 3-dim velocity command."""
        return torch.stack([self._vx, self._vy, self._yaw], dim=1)

    @property
    def ref_joint_pos(self) -> torch.Tensor:
        """Reference joint positions at current frame, shape (N, J)."""
        return self._ref_joint_pos[self._time_step]

    @property
    def ref_joint_vel(self) -> torch.Tensor:
        """Reference joint velocities at current frame, shape (N, J)."""
        return self._ref_joint_vel[self._time_step]

    @property
    def ref_anchor_pos_w(self) -> torch.Tensor:
        """Reference root position in world frame, shape (N, 3)."""
        return self._ref_root_pos[self._time_step] + self._env.scene.env_origins

    @property
    def ref_anchor_quat_w(self) -> torch.Tensor:
        """Reference root orientation in world frame, shape (N, 4)."""
        return self._ref_root_quat[self._time_step]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Robot's current root position in world frame, shape (N, 3)."""
        return self.robot.data.root_pos_w

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Robot's current root orientation in world frame, shape (N, 4)."""
        return self.robot.data.root_quat_w

    @property
    def cmd_vx(self) -> torch.Tensor:
        return self._vx

    @property
    def cmd_vy(self) -> torch.Tensor:
        return self._vy

    @property
    def cmd_yaw(self) -> torch.Tensor:
        return self._yaw

    # ── Core logic ─────────────────────────────────────────────

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands and reset to a random start frame."""
        if len(env_ids) == 0:
            return

        # Uniform sample velocity commands
        vx = sample_uniform(
            self.cfg.vx_min, self.cfg.vx_max,
            (len(env_ids),), device=self.device,
        )
        vy = sample_uniform(
            self.cfg.vy_min, self.cfg.vy_max,
            (len(env_ids),), device=self.device,
        )
        yaw = sample_uniform(
            self.cfg.yaw_min, self.cfg.yaw_max,
            (len(env_ids),), device=self.device,
        )

        # Zero-out tiny commands to avoid no-op episodes
        speed = torch.sqrt(vx ** 2 + vy ** 2)
        zero_mask = speed < self.cfg.min_command_speed
        vx[zero_mask] = 0.0
        vy[zero_mask] = 0.0

        self._vx[env_ids] = vx
        self._vy[env_ids] = vy
        self._yaw[env_ids] = yaw

        # Always start from frame 0 (balanced walk-cycle pose)
        self._time_step[env_ids] = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)

    def _update_command(self):
        """Advance one frame per step (no velocity scaling in V1)."""
        self._time_step += 1

        over = torch.where(self._time_step >= self._n_frames)[0]
        if len(over) > 0:
            self._resample_command(over)

    def _update_metrics(self):
        robot_lin = self.robot.data.root_lin_vel_b[:, :2]   # (N, 2)
        robot_yaw = self.robot.data.root_ang_vel_b[:, 2]    # (N,)

        self.metrics["vx_err"] = torch.abs(robot_lin[:, 0] - self._vx)
        self.metrics["vy_err"] = torch.abs(robot_lin[:, 1] - self._vy)
        self.metrics["yaw_err"] = torch.abs(robot_yaw - self._yaw)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class VelocityCommandCfg(CommandTermCfg):
    """Configuration for the VelocityCommand term."""

    class_type: type = VelocityCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    motion_file: str = MISSING
    """Path to the NPZ reference motion clip."""

    # ── Velocity command range ────────────────────────────────
    vx_min: float = -1.0
    vx_max: float = 1.0
    vy_min: float = -0.4
    vy_max: float = 0.4
    yaw_min: float = -0.8
    yaw_max: float = 0.8

    min_command_speed: float = 0.05
    """Commands below this speed are zeroed to avoid drift."""

    # ── Reference playback ────────────────────────────────────
    ref_forward_speed: float = 0.6
    """Forward velocity (m/s) of the reference clip; used for speed scaling."""
