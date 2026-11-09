"""Replay a trained whole-body-tracking checkpoint in native MuJoCo.

This is a cross-simulator evaluation of the actor actually learned in Isaac Lab:
the observation is rebuilt with the exact layout used by the tracking task
(160 dims: reference joint pos/vel, relative anchor pos/orientation, base
linear/angular velocity, joint pos/vel, last action), the checkpoint's
empirical normalizer is applied, and PD torques drive the MuJoCo model.
No reference playback, pose correction or reset hides tracking failures.

The policy joint order is the Isaac Lab articulation order (the interleaved
left/right order observed by training), not the SDK order of the CSV/MJCF.

Supported robots (``--robot``, default ``g1``): ``g1`` (interleaved articulation
order, actuators present in the holosoma MJCF) and ``f1`` (weichai_f1_29 URDF
document order; the UMR MJCF has no actuators, so torques go through
``qfrc_applied``).

Example::

    python scripts/replay_mujoco_policy.py \
        --checkpoint ../czy/data/g1_exp1/model_2999.pt \
        --motion source/whole_body_tracking/whole_body_tracking/assets/g1/motions/dance2_subject5_22s.npz \
        --xml ../holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml \
        --video ../czy/data/g1_exp1/mujoco_replay_2999.mp4 \
        --report ../czy/analysis/g1_exp1_curves/mujoco_replay_2999.json

    python scripts/replay_mujoco_policy.py --robot f1 \
        --checkpoint czy/data/f1_walk_b3/model_2999.pt \
        --motion source/whole_body_tracking/whole_body_tracking/assets/f1/motions/converted/B3_walk1_weichai_f1_29.npz \
        --xml source/whole_body_tracking/whole_body_tracking/assets/f1/mjcf/f1_29dof_scene.xml \
        --video czy/data/f1_walk_b3/mujoco_replay_2999.mp4 \
        --report czy/data/f1_walk_b3/mujoco_replay_2999.json
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Robot profiles. POLICY_JOINT_ORDER is the Isaac Lab articulation order used
# for the policy's action/observation vectors, not the SDK/MJCF joint order.
# ---------------------------------------------------------------------------

G1_POLICY_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# F1 (weichai_f1_29, SYS-SPEC-0002-V1.0 URDF). This is the exact order Isaac Lab
# reports for the URDF-spawned articulation (verified on the training image with
# scripts/debug_joint_order.py, TASK_20261108_045): the importer interleaves
# left / spine / right chains (left_hip_pitch, lumbar_yaw, right_hip_pitch, ...),
# NOT the URDF document order.
F1_POLICY_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "lumbar_yaw_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "lumbar_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "lumbar_pitch_joint",
    "right_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "right_ankle_roll_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

# Bodies tracked by the F1 tracking task (config/f1/flat_env_cfg.py body_names).
F1_TRACKED_BODY_NAMES = [
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

ROBOTS_DIR = Path(__file__).resolve().parents[1] / "source/whole_body_tracking/whole_body_tracking/robots"

ROBOT_PROFILES = {
    "g1": {
        "policy_joint_order": G1_POLICY_JOINT_ORDER,
        "tracked_body_names": G1_TRACKED_BODY_NAMES,
        "anchor_body": "torso_link",
        "root_body": "pelvis",
        "robot_config": ROBOTS_DIR / "g1.py",
        "cfg_constant": "G1_CYLINDER_CFG",
        "pose_constant": None,
        "action_scale_constant": None,
    },
    "f1": {
        "policy_joint_order": F1_POLICY_JOINT_ORDER,
        "tracked_body_names": F1_TRACKED_BODY_NAMES,
        "anchor_body": "base_link",
        "root_body": "base_link",
        "robot_config": ROBOTS_DIR / "f1.py",
        "cfg_constant": "F1_CFG",
        "pose_constant": "F1_DEFAULT_JOINT_POS",
        "action_scale_constant": "F1_ACTION_SCALE",
        "disable_self_collision": True,
    },
}

FALL_HEIGHT = 0.35
FALL_TILT = 0.85
NORMALIZER_EPS = 0.01


# ---------------------------- quaternion helpers ----------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_mat(quat_conj(q)) @ v


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_error(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(a, b)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


# --------------------------- training parameters ---------------------------

def _assignment(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise ValueError(f"missing training definition: {name}")


def _evaluate(node, constants):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(n, constants) for n in node.elts]
    if isinstance(node, ast.Dict):
        return {_evaluate(k, constants): _evaluate(v, constants) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand, constants)
    if isinstance(node, ast.BinOp):
        op = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}[type(node.op)]
        return op(_evaluate(node.left, constants), _evaluate(node.right, constants))
    raise ValueError(f"unsupported training expression: {ast.dump(node)}")


def _kwargs(call):
    return {k.arg: k.value for k in call.keywords}


def load_training_parameters(
    robot_py: Path,
    names: list[str],
    cfg_constant: str = "G1_CYLINDER_CFG",
    pose_constant: str | None = None,
    action_scale_constant: str | None = None,
) -> dict[str, np.ndarray]:
    """Read default pose, PD gains, effort limits and action scale from robots/<robot>.py."""
    tree = ast.parse(robot_py.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = _evaluate(node.value, constants)
            except (KeyError, ValueError, TypeError):
                pass
    robot = _kwargs(_assignment(tree, cfg_constant))
    if pose_constant is not None:
        default = _evaluate(_assignment(tree, pose_constant), constants)
    else:
        default = _evaluate(_kwargs(robot["init_state"])["joint_pos"], constants)
    actuators = [_kwargs(v) for v in robot["actuators"].values]
    if action_scale_constant is not None:
        action_scale = _evaluate(_assignment(tree, action_scale_constant), constants)
    else:
        action_scale = None
    out = {key: [] for key in ("default_joint_pos", "kp", "kd", "effort_limit", "action_scale")}
    for name in names:
        matches = [a for a in actuators if any(re.fullmatch(p, name) for p in _evaluate(a["joint_names_expr"], constants))]
        if len(matches) != 1:
            raise ValueError(f"ambiguous actuator for {name}")
        values = {}
        for key in ("stiffness", "damping", "effort_limit_sim"):
            value = _evaluate(matches[0][key], constants)
            if isinstance(value, dict):
                candidates = [v for p, v in value.items() if re.fullmatch(p, name)]
                if len(candidates) != 1:
                    raise ValueError(f"ambiguous {key} for {name}")
                value = candidates[0]
            values[key] = float(value)
        out["default_joint_pos"].append(
            next((float(v) for p, v in default.items() if re.fullmatch(p, name)), 0.0)
        )
        out["kp"].append(values["stiffness"])
        out["kd"].append(values["damping"])
        out["effort_limit"].append(values["effort_limit_sim"])
        if action_scale is None:
            out["action_scale"].append(0.25 * values["effort_limit_sim"] / values["stiffness"])
        elif isinstance(action_scale, dict):
            candidates = [v for p, v in action_scale.items() if re.fullmatch(p, name)]
            if len(candidates) != 1:
                raise ValueError(f"ambiguous action scale for {name}")
            out["action_scale"].append(float(candidates[0]))
        else:
            out["action_scale"].append(float(action_scale))
    return {key: np.asarray(value, dtype=np.float64) for key, value in out.items()}


# -------------------------------- policy -----------------------------------

class Actor(nn.Module):
    def __init__(self, checkpoint: dict):
        super().__init__()
        state = checkpoint["model_state_dict"]
        ids = sorted(int(k.split(".")[1]) for k in state if re.fullmatch(r"actor\.\d+\.weight", k))
        layers = []
        for i, idx in enumerate(ids):
            w, b = state[f"actor.{idx}.weight"], state[f"actor.{idx}.bias"]
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)
            layers.append(layer)
            if i + 1 != len(ids):
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        norm = checkpoint["obs_norm_state_dict"]
        self.register_buffer("mean", norm["_mean"].float().clone())
        self.register_buffer("std", norm["_std"].float().clone())
        self.eps = NORMALIZER_EPS
        if self.mean.numel() != self.actor[0].in_features:
            raise ValueError("normalizer and actor dimensions differ")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor((obs - self.mean) / (self.std + self.eps))


def load_checkpoint(path: Path) -> tuple[Actor, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(checkpoint).eval()
    iteration = int(checkpoint.get("iter", -1))
    return actor, iteration


# -------------------------------- replay -----------------------------------

class Replay:
    def __init__(self, args: argparse.Namespace) -> None:
        torch.set_num_threads(1)
        self.profile = ROBOT_PROFILES[args.robot]
        policy_joint_order = self.profile["policy_joint_order"]
        tracked_body_names = self.profile["tracked_body_names"]
        self.actor, self.iteration = load_checkpoint(args.checkpoint)
        if self.actor.actor[-1].out_features != len(policy_joint_order):
            raise ValueError(f"checkpoint action dimension does not match the {args.robot} policy")

        with np.load(args.motion, allow_pickle=False) as motion:
            self.motion = {key: motion[key].copy() for key in motion.files}
        if "joint_names" not in self.motion or "body_names" not in self.motion:
            raise ValueError("motion npz needs joint_names/body_names")
        self.fps = float(np.asarray(self.motion["fps"]).flat[0])
        if abs(self.fps - 50.0) > 1e-6:
            raise ValueError("this task requires a 50 Hz reference motion")
        motion_joint_names = [str(n) for n in self.motion["joint_names"]]
        motion_body_names = [str(n) for n in self.motion["body_names"]]
        missing = [n for n in policy_joint_order if n not in motion_joint_names]
        if missing:
            raise ValueError(f"motion is missing policy joints: {missing}")
        self.motion_joint_ids = np.array([motion_joint_names.index(n) for n in policy_joint_order])
        self.torso_ref_id = motion_body_names.index(self.profile["anchor_body"])
        self.root_ref_id = motion_body_names.index(self.profile["root_body"])
        self.tracked_ref_ids = np.array([motion_body_names.index(n) for n in tracked_body_names])
        self.frames = int(self.motion["joint_pos"].shape[0])

        params = load_training_parameters(
            args.robot_config,
            policy_joint_order,
            cfg_constant=self.profile["cfg_constant"],
            pose_constant=self.profile["pose_constant"],
            action_scale_constant=self.profile["action_scale_constant"],
        )
        self.default = params["default_joint_pos"]
        self.kp = params["kp"] * args.kp_scale
        self.kd = params["kd"] * args.kd_scale
        self.effort = params["effort_limit"]
        self.action_scale = params["action_scale"]

        self.model = mujoco.MjModel.from_xml_path(str(args.xml))
        if self.profile.get("disable_self_collision", False):
            # The URDF-converted MJCF marks every collision proxy contype=1/conaffinity=1 and
            # carries no <contact><exclude> pairs; adjacent link meshes overlap by centimetres
            # at the nominal pose and produce huge spurious contacts (the legs get kicked apart).
            # Keep robot-ground contact only: robot geoms (2,1) vs world geoms (1,2) collide,
            # robot-robot (2 vs 1) does not.
            world = self.model.geom_bodyid == 0
            self.model.geom_contype[~world] = 2
            self.model.geom_conaffinity[~world] = 1
            self.model.geom_contype[world] = 1
            self.model.geom_conaffinity[world] = 2
        if self.model.nq != 7 + len(policy_joint_order):
            raise ValueError(f"unexpected MJCF nq={self.model.nq}")
        self.model.opt.timestep = args.timestep
        self.decimation = max(1, int(round(args.control_dt / args.timestep)))
        self.dt = self.model.opt.timestep * self.decimation
        if abs(self.dt - 1.0 / self.fps) > 1e-9:
            raise ValueError("policy and motion clocks differ")

        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in policy_joint_order]
        if min(joints) < 0:
            raise ValueError("MJCF is missing policy joints")
        self.qpos_ids = np.array([self.model.jnt_qposadr[j] for j in joints])
        self.qvel_ids = np.array([self.model.jnt_dofadr[j] for j in joints])
        # Actuator-driven MJCFs (G1) address data.ctrl; actuatorless MJCFs (F1) get
        # the same PD torque through data.qfrc_applied on the joint dof.
        actuators = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in policy_joint_order]
        self.use_ctrl = bool(np.all(np.array(actuators) >= 0))
        self.actuator_ids = np.array(actuators) if self.use_ctrl else None
        if not self.use_ctrl:
            missing_actuators = [n for n, a in zip(policy_joint_order, actuators) if a < 0]
            if len(missing_actuators) != len(policy_joint_order):
                raise ValueError(f"MJCF has actuators for only some policy joints: {missing_actuators[:3]}...")
        self.lower = np.array([self.model.jnt_range[j, 0] for j in joints])
        self.upper = np.array([self.model.jnt_range[j, 1] for j in joints])
        self.root_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.profile["root_body"])
        self.torso_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.profile["anchor_body"])
        self.tracked_bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in tracked_body_names]
        if min([self.root_body, self.torso_body, *self.tracked_bodies]) < 0:
            raise ValueError("MJCF is missing tracked bodies")

        self.data = mujoco.MjData(self.model)
        self.repeat = args.repeat_motion
        self.deploy_limits = args.deploy_limits
        self.action = np.zeros(len(policy_joint_order))
        self.index = args.start_frame
        self.last_target = None
        self.metrics: list[dict] = []
        self.reset()

    # -- state helpers ------------------------------------------------------
    def reference_frame(self, index: int) -> int:
        return index % self.frames if self.repeat else min(index, self.frames - 1)

    def reset(self) -> None:
        frame = self.reference_frame(self.index)
        root_pos = self.motion["body_pos_w"][frame, self.root_ref_id]
        root_quat = self.motion["body_quat_w"][frame, self.root_ref_id]
        root_lin = self.motion["body_lin_vel_w"][frame, self.root_ref_id]
        root_ang = self.motion["body_ang_vel_w"][frame, self.root_ref_id]
        joint_pos = self.motion["joint_pos"][frame, self.motion_joint_ids]
        joint_vel = self.motion["joint_vel"][frame, self.motion_joint_ids]
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[self.qpos_ids] = joint_pos
        self.data.qvel[:3] = root_lin
        self.data.qvel[3:6] = quat_to_mat(root_quat).T @ root_ang
        self.data.qvel[self.qvel_ids] = joint_vel
        mujoco.mj_forward(self.model, self.data)
        self.action[:] = 0.0
        self.last_target = joint_pos.copy()

    def observation(self, frame: int) -> np.ndarray:
        ref_q = self.motion["joint_pos"][frame, self.motion_joint_ids]
        ref_dq = self.motion["joint_vel"][frame, self.motion_joint_ids]
        ref_torso_pos = self.motion["body_pos_w"][frame, self.torso_ref_id]
        ref_torso_quat = self.motion["body_quat_w"][frame, self.torso_ref_id]
        robot_torso_pos = self.data.xpos[self.torso_body].copy()
        robot_torso_quat = self.data.xquat[self.torso_body].copy()

        relative_pos = quat_rotate_inverse(robot_torso_quat, ref_torso_pos - robot_torso_pos)
        relative_rot = quat_to_mat(quat_mul(quat_conj(robot_torso_quat), ref_torso_quat))
        relative_ori = relative_rot[:, :2].reshape(-1)

        root_rot = quat_to_mat(self.data.qpos[3:7])
        linear_velocity_b = root_rot.T @ self.data.qvel[:3]
        angular_velocity_b = self.data.qvel[3:6]
        joint_pos = self.data.qpos[self.qpos_ids] - self.default
        joint_vel = self.data.qvel[self.qvel_ids]

        return np.concatenate([
            ref_q, ref_dq, relative_pos, relative_ori,
            linear_velocity_b, angular_velocity_b,
            joint_pos, joint_vel, self.action,
        ]).astype(np.float32)

    # -- control ------------------------------------------------------------
    def make_target(self, action: np.ndarray) -> np.ndarray:
        desired = self.default + self.action_scale * action
        if not self.deploy_limits:
            return desired
        limited = np.clip(desired, self.lower + 0.01, self.upper - 0.01)
        speed = 8.0
        limited = np.clip(limited, self.last_target - speed * self.dt, self.last_target + speed * self.dt)
        center = self.data.qpos[self.qpos_ids] + self.kd * self.data.qvel[self.qvel_ids] / self.kp
        radius = self.effort / self.kp
        limited = np.clip(limited, center - radius, center + radius)
        return np.clip(limited, self.lower + 0.01, self.upper - 0.01)

    def control_step(self) -> None:
        frame = self.reference_frame(self.index)
        obs = self.observation(frame)
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
        if not np.isfinite(action).all():
            raise RuntimeError("policy produced a non-finite action")
        target = self.make_target(action)
        for _ in range(self.decimation):
            torque = self.kp * (target - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
            torque = np.clip(torque, -self.effort, self.effort)
            if self.use_ctrl:
                self.data.ctrl[self.actuator_ids] = torque
            else:
                self.data.qfrc_applied[:] = 0.0
                self.data.qfrc_applied[self.qvel_ids] = torque
            mujoco.mj_step(self.model, self.data)
        self.action[:] = action
        self.last_target = target
        self.index += 1
        mujoco.mj_forward(self.model, self.data)

    def record(self) -> None:
        frame = self.reference_frame(self.index)
        tracked_pos = self.data.xpos[self.tracked_bodies]
        tracked_quat = self.data.xquat[self.tracked_bodies]
        ref_pos = self.motion["body_pos_w"][frame, self.tracked_ref_ids]
        ref_quat = self.motion["body_quat_w"][frame, self.tracked_ref_ids]
        pos_errors = np.linalg.norm(tracked_pos - ref_pos, axis=1)
        rot_errors = np.array([quat_error(tracked_quat[i], ref_quat[i]) for i in range(len(self.tracked_bodies))])
        root_rot = quat_to_mat(self.data.qpos[3:7])
        tilt = float(np.degrees(np.arccos(np.clip(root_rot[2, 2], -1.0, 1.0))))
        self.metrics.append({
            "t": round(self.index * self.dt, 3),
            "reference_frame": frame,
            "root_x": round(float(self.data.qpos[0]), 4),
            "root_y": round(float(self.data.qpos[1]), 4),
            "height": round(float(self.data.qpos[2]), 4),
            "tilt_deg": round(tilt, 2),
            "anchor_pos_error_m": round(float(np.linalg.norm(tracked_pos[7] - ref_pos[7])), 4),
            "anchor_rot_error_rad": round(quat_error(tracked_quat[7], ref_quat[7]), 4),
            "body_pos_mean_m": round(float(pos_errors.mean()), 4),
            "body_pos_max_m": round(float(pos_errors.max()), 4),
            "body_rot_mean_rad": round(float(rot_errors.mean()), 4),
            "body_rot_max_rad": round(float(rot_errors.max()), 4),
            "joint_pos_error_rad": round(
                float(np.linalg.norm(
                    self.data.qpos[self.qpos_ids]
                    - self.motion["joint_pos"][frame, self.motion_joint_ids]
                )), 4
            ),
            "action_abs_max": round(float(np.abs(self.action).max()), 4),
        })

    def fell(self) -> bool:
        root_rot = quat_to_mat(self.data.qpos[3:7])
        tilt = np.arccos(np.clip(root_rot[2, 2], -1.0, 1.0))
        return bool(self.data.qpos[2] < FALL_HEIGHT or tilt > FALL_TILT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_PROFILES),
        default="g1",
        help="Robot profile: policy joint/body order and default robot config.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="Robot config python file (default: robots/<robot>.py).",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=None, help="default: the whole reference clip")
    parser.add_argument("--repeat-motion", action="store_true")
    parser.add_argument("--deploy-limits", action="store_true",
                        help="also apply the deployment target limiter (speed/effort/limits)")
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-azimuth", type=float, default=135.0, help="Camera compass azimuth (deg).")
    parser.add_argument("--camera-elevation", type=float, default=-14.0, help="Camera elevation (deg).")
    parser.add_argument("--camera-distance", type=float, default=2.6, help="Camera distance (m).")
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Render the reference clip (kinematic playback) next to the policy, same camera.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--kp-scale", type=float, default=1.0,
                        help="Global stiffness (Kp) multiplier (default: 1.0).")
    parser.add_argument("--kd-scale", type=float, default=1.0,
                        help="Global damping (Kd) multiplier (default: 1.0).")
    args = parser.parse_args()
    if args.robot_config is None:
        args.robot_config = ROBOT_PROFILES[args.robot]["robot_config"]

    replay = Replay(args)
    steps = replay.frames - args.start_frame
    if args.seconds is not None:
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        steps = min(steps, int(round(args.seconds / replay.dt)))
    if steps <= 0:
        parser.error("nothing to replay")

    started = time.perf_counter()
    writer = renderer = camera = None
    raw_video = None
    ref_data = None
    if args.video is not None:
        import cv2

        args.video.parent.mkdir(parents=True, exist_ok=True)
        raw_video = args.video.with_suffix(".raw.mp4")
        replay.model.vis.global_.offwidth = args.width
        replay.model.vis.global_.offheight = args.height
        renderer = mujoco.Renderer(replay.model, height=args.height, width=args.width)
        writer_width = args.width * 2 if args.compare_reference else args.width
        writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.video_fps, (writer_width, args.height))
        if not writer.isOpened():
            raise RuntimeError("video writer did not open")
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = (
            args.camera_distance,
            args.camera_azimuth,
            args.camera_elevation,
        )
        if args.compare_reference:
            ref_data = mujoco.MjData(replay.model)

    print(
        "MUJOCO_REPLAY_START",
        json.dumps({
            "iteration": replay.iteration,
            "frames": steps,
            "duration_s": round(steps * replay.dt, 3),
            "reference_frames": replay.frames,
            "deploy_limits": args.deploy_limits,
            "repeat_motion": args.repeat_motion,
        }),
        flush=True,
    )
    try:
        for _ in range(steps):
            if args.repeat_motion and replay.index > 0 and replay.index % replay.frames == 0:
                replay.reset()
            replay.record()
            if writer is not None:
                import cv2

                camera.lookat[:] = replay.data.xpos[replay.root_body]
                camera.lookat[2] = 0.8
                renderer.update_scene(replay.data, camera=camera)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                if ref_data is not None:
                    frame_index = replay.reference_frame(replay.index)
                    ref_data.qpos[:] = 0.0
                    ref_data.qpos[:3] = replay.motion["body_pos_w"][frame_index, replay.root_ref_id]
                    ref_data.qpos[3:7] = replay.motion["body_quat_w"][frame_index, replay.root_ref_id]
                    ref_data.qpos[replay.qpos_ids] = replay.motion["joint_pos"][frame_index, replay.motion_joint_ids]
                    mujoco.mj_forward(replay.model, ref_data)
                    camera.lookat[:] = ref_data.xpos[replay.root_body]
                    camera.lookat[2] = 0.8
                    renderer.update_scene(ref_data, camera=camera)
                    reference_panel = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                    cv2.putText(reference_panel, "reference", (20, args.height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"policy (MuJoCo) | iter {replay.iteration}", (20, args.height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    frame = cv2.hconcat([reference_panel, frame])
                    cv2.line(frame, (args.width, 0), (args.width, args.height), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"{args.robot.upper()} BeyondMimic | iter {replay.iteration} | t={replay.index * replay.dt:.2f}s "
                    f"| frame {replay.reference_frame(replay.index)}",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                writer.write(frame)
            replay.control_step()
            if replay.fell():
                print("MUJOCO_REPLAY_FALL", json.dumps({"time_s": round(replay.index * replay.dt, 3)}), flush=True)
                break
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()

    if raw_video is not None:
        import subprocess

        try:
            import imageio_ffmpeg

            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-i", str(raw_video),
                 "-an", "-c:v", "libx264", "-crf", "20", "-preset", "fast",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.video)],
                check=True, capture_output=True, timeout=300,
            )
            raw_video.unlink(missing_ok=True)
        except Exception as error:  # keep the working mp4v file if ffmpeg is unavailable
            print(f"WARN: ffmpeg conversion failed ({error}); keeping {raw_video}", flush=True)
            raw_video.replace(args.video)

    metrics = replay.metrics
    position_errors = [m["body_pos_mean_m"] for m in metrics]
    orientation_errors = [m["body_rot_mean_rad"] for m in metrics]
    report = {
        "robot": args.robot,
        "checkpoint": str(args.checkpoint),
        "iteration": replay.iteration,
        "motion": str(args.motion),
        "mujoco": mujoco.__version__,
        "physics_dt": replay.model.opt.timestep,
        "control_dt": replay.dt,
        "reference_frames": replay.frames,
        "replayed_frames": len(metrics),
        "simulated_seconds": round(len(metrics) * replay.dt, 3),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "deploy_limits": args.deploy_limits,
        "fell": replay.fell(),
        "min_height_m": min(m["height"] for m in metrics),
        "max_tilt_deg": max(m["tilt_deg"] for m in metrics),
        "body_pos_mean_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))),
        "body_pos_mean_max_m": max(position_errors),
        "body_rot_mean_max_rad": max(orientation_errors),
        "anchor_pos_error_max_m": max(m["anchor_pos_error_m"] for m in metrics),
        "joint_pos_error_max_rad": max(m["joint_pos_error_rad"] for m in metrics),
        "video": str(args.video) if args.video else None,
        "trajectory": metrics,
    }
    print("MUJOCO_REPLAY_RESULT", json.dumps({k: v for k, v in report.items() if k != "trajectory"}), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["fell"]:
        print("REPLAY_OK: robot stayed upright for the whole segment")


if __name__ == "__main__":
    main()
