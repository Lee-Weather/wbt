"""
Termination functions for velocity-conditioned locomotion.

Uses robot state directly (no reference-trajectory dependency) so it works
with VelocityCommand.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_fall(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.30,
) -> torch.Tensor:
    """Terminate if the robot's base (pelvis) drops below threshold.

    Works with any command type — only needs the robot's root height.
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # TEMP DIAGNOSTIC (remove after debugging the instant-termination issue)
    n_dbg = getattr(env, "_dbg_n", 0)
    if n_dbg < 8:
        env._dbg_n = n_dbg + 1
        buf = env.episode_length_buf
        print(
            f"[DBG#{n_dbg}] ctr={env.common_step_counter}"
            f" max_ep_len={getattr(env, 'max_episode_length', 'NA')}"
            f" buf_min={int(buf.min())} buf_max={int(buf.max())}"
            f" terms={env.termination_manager.active_terms}"
            f" z_mean={asset.data.root_pos_w[:, 2].mean().item():.3f}"
            f" frac={float((height < threshold).float().mean()):.3f}",
            flush=True,
        )
    return height < threshold


def base_tilt(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if the robot's base orientation is too tilted.

    Projects gravity into the robot's base frame and checks the vertical
    component — a highly tilted base will have a small vertical component.
    """
    asset = env.scene[asset_cfg.name]
    # gravity vector in world frame (usually [0, 0, -9.81] or just direction)
    gravity_dir = asset.data.GRAVITY_VEC_W  # (3,) — unit direction
    # rotate gravity into robot's base frame
    gravity_base = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w, gravity_dir
    )
    # vertical component of gravity in base frame
    # if upright: gravity_base[:, 2] ≈ -1  →  cos_angle ≈ 1
    cos_angle = -gravity_base[:, 2]
    return cos_angle < torch.cos(torch.tensor(threshold))


def limbs_below_ground(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=[
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
    ),
    threshold: float = 0.10,
) -> torch.Tensor:
    """Terminate if limb bodies drop below ground level.

    This catches the robot's feet/hands clipping through the floor.
    """
    asset = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    height = body_pos[:, :, 2] - env.scene.env_origins[:, None, 2]
    return torch.any(height < -threshold, dim=-1)
