"""
Action terms that use the reference joint positions from the velocity command
as the action baseline, so the policy only learns a residual (small offset).

This directly addresses the non-convergence problem:
  target_joint_pos = ref_joint_pos[t] + policy_action * small_scale

With this approach, even random policy outputs produce a walking-like motion
because the reference trajectory provides the core gait. The policy simply
modulates around it to track the commanded velocity.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass, string_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Action Term ──────────────────────────────────────────────────────


class RefJointPositionAction(ActionTerm):
    """Joint position action using reference trajectory as baseline.

    In each step, this reads the reference joint positions from the command term
    and computes:
        joint_target = ref_joint_pos + policy_action * scale

    This ensures the robot always tracks the reference gait, while the policy
    only needs to learn an offset to adjust speed / direction.
    """

    is_graphable: bool = False  # reads from external command term

    cfg: RefJointPositionActionCfg

    def __init__(self, cfg: RefJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Resolve asset and joint indices
        self._asset: Articulation = env.scene[cfg.asset_name]

        joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names,
            preserve_order=cfg.preserve_order,
            as_proxy=False,
        )
        self._num_joints = len(joint_ids)

        # If all joints are covered, use slice for efficiency
        if self._num_joints == self._asset.num_joints and not cfg.preserve_order:
            self._joint_ids = slice(None)
        else:
            self._joint_ids = joint_ids.to(self._asset.device)

        # Scale is a scalar applied uniformly to all joints
        self._scale = cfg.scale
        self._command_name = cfg.command_name

        # Buffers
        device = self._device
        num_envs = self._num_envs
        self._raw_actions = torch.zeros(num_envs, self._num_joints, device=device)
        self._processed_actions = torch.zeros(num_envs, self._num_joints, device=device)

    # ── Properties ─────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ── Core logic ─────────────────────────────────────────────────

    def process_actions(self, actions: torch.Tensor):
        """Compute joint targets as ref_joint_pos + actions * scale."""
        self._raw_actions = actions.clone()

        # Read reference joint positions from the velocity command
        cmd = self._env.command_manager.get_term(self._command_name)
        ref_pos = cmd.ref_joint_pos  # (N, J)

        # target = ref_pos + actions * scale
        processed = ref_pos + actions * self._scale

        # Apply clipping if configured (dict[str, tuple] → per-joint tensor)
        if self.cfg.clip is not None:
            clip_low = []
            clip_high = []
            for name in self._joint_names:
                matched = False
                for pattern, (lo, hi) in self.cfg.clip.items():
                    if string_utils.matches_pattern(name, pattern):
                        clip_low.append(lo)
                        clip_high.append(hi)
                        matched = True
                        break
                if not matched:
                    clip_low.append(-torch.inf)
                    clip_high.append(torch.inf)
            clip_low = torch.tensor(clip_low, device=processed.device).unsqueeze(0)
            clip_high = torch.tensor(clip_high, device=processed.device).unsqueeze(0)
            processed = torch.clamp(processed, clip_low, clip_high)

        self._processed_actions = processed

    def reset(self, env_ids: list[int] | None = None):
        """Reset the action term (no-op for this implementation)."""
        pass

    def apply_actions(self):
        """Send computed joint positions to the simulator."""
        self._asset.set_joint_position_target(
            target=self._processed_actions,
            joint_ids=self._joint_ids,
        )

    # ── Debug visualisation (unused) ──────────────────────────────

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


# ── Config ───────────────────────────────────────────────────────────


@configclass
class RefJointPositionActionCfg(ActionTermCfg):
    """Configuration for the reference-guided joint position action term.

    The policy outputs a residual offset around the reference joint positions
    from the velocity command, rather than learning the full action from scratch.

    Attributes:
        command_name: Name of the VelocityCommand term providing the reference.
        scale: Scalar multiplier applied to the policy output (residual).
        joint_names: Which joints to control. Supports glob patterns like ``.*``.
        preserve_order: Whether to preserve the order of joint names after resolution.
    """
    class_type: type = RefJointPositionAction
    command_name: str = "motion"
    scale: float = 0.05
    joint_names: list[str] | None = None
    preserve_order: bool = True
