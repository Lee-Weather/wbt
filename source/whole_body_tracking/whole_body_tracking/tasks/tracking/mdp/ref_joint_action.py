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
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation


class RefJointPositionAction(ActionTerm):
    """Joint position action using reference trajectory as baseline.

    In each step, this reads the reference joint positions from the velocity command
    and computes:
        joint_target = ref_joint_pos + policy_action * scale

    This ensures the robot always follows a walking gait (from the reference),
    while the policy only learns small modulations to track the commanded velocity.
    """

    is_graphable: bool = False  # reads from external command term

    cfg: RefJointPositionActionCfg

    def __init__(self, cfg: RefJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._command_name = cfg.command_name
        self._num_joints = self._asset.num_joints
        self._scale = cfg.scale

        # Buffers: (num_envs, num_joints)
        # NOTE: ActionTerm does not define `_device` / `_num_envs`; use the
        # ManagerTermBase properties (same pattern as VelocityCommand).
        device = self.device
        n = self.num_envs
        self._raw_actions = torch.zeros(n, self._num_joints, device=device)
        self._processed_actions = torch.zeros(n, self._num_joints, device=device)

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
        self._processed_actions = ref_pos + actions * self._scale

    def reset(self, env_ids: list[int] | None = None):
        pass

    def apply_actions(self):
        """Send joint targets to the physics engine."""
        self._asset.set_joint_position_target(self._processed_actions)

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
    """
    class_type: type = RefJointPositionAction
    command_name: str = "motion"
    scale: float = 0.1
