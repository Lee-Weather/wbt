"""Convert a retargeted motion (GMR pkl or qpos npz) into the npz format used by motion tracking.

Supported inputs:
  * .pkl : fps, root_pos (T,3), root_rot (T,4, xyzw), dof_pos (T,J), dof_names
  * .npz : fps, qpos (T,3+4+J, wxyz), robot_joint_names

The motion is resampled to 50 Hz (the tracking policy rate), centered on the XY plane, and
replayed in Isaac Sim so that joint order matches the articulation exactly.

.. code-block:: bash

    # Usage
    python pkl_to_npz.py \
        --input_file source/whole_body_tracking/whole_body_tracking/assets/x1/motions/accad_120hz/B3_walk1_x1.npz \
        --output_file source/whole_body_tracking/whole_body_tracking/assets/x1/motions/B3_walk1_x1_converted.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pickle
import sys
from pathlib import Path

# make the extension importable without `pip install -e` (e.g. on the training platform)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"))

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert a retargeted motion to the tracking npz format.")
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the input .pkl or qpos .npz file, or a directory containing such files (batch mode).",
)
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Path to the output npz file, or a directory when --input_file is a directory.",
)
parser.add_argument("--max_frames", type=int, default=-1, help="Optionally truncate the output to this many frames.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

##
# Pre-defined configs
##
from whole_body_tracking.robots.x1 import X1_CFG

# The tracking policy runs at 50 Hz (sim.dt 0.005 * decimation 4), so the reference must match.
TRACKING_FPS = 50.0


@configclass
class ConvertSceneCfg(InteractiveSceneCfg):
    """Scene for replaying a reference motion without physics."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = X1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation between two batches of (wxyz) quaternions."""
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.clamp(torch.abs(dot), max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta < 1e-6
    w0 = torch.where(small, 1.0 - blend, torch.sin((1.0 - blend) * theta) / torch.clamp(sin_theta, min=1e-6))
    w1 = torch.where(small, blend, torch.sin(blend * theta) / torch.clamp(sin_theta, min=1e-6))
    out = w0 * q0 + w1 * q1
    return out / torch.linalg.norm(out, dim=-1, keepdim=True)


class MotionLoader:
    """Loads a retargeted motion and resamples it to the tracking rate."""

    def __init__(self, motion_file: str, device: torch.device):
        if motion_file.endswith(".pkl"):
            self._load_pkl(motion_file, device)
        elif motion_file.endswith(".npz"):
            self._load_qpos_npz(motion_file, device)
        else:
            raise ValueError(f"Unsupported motion file: {motion_file}")

        if abs(self.fps - TRACKING_FPS) > 1e-6:
            self._resample(device)

        # Center the motion on the XY plane using the first frame.
        self.root_pos[:, :2] -= self.root_pos[0, :2]

        self.dt = 1.0 / TRACKING_FPS
        self.dof_vel = torch.gradient(self.dof_pos, spacing=self.dt, dim=0)[0]
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_quat, self.dt)
        self.num_frames = self.dof_pos.shape[0]

    def _load_pkl(self, motion_file: str, device: torch.device):
        with open(motion_file, "rb") as f:
            data = pickle.load(f)
        self.fps = float(data["fps"])
        self.dof_names = list(data["dof_names"])
        self.root_pos = torch.tensor(data["root_pos"], dtype=torch.float32, device=device)
        # GMR pkl stores quaternions as xyzw, Isaac Lab uses wxyz.
        self.root_quat = torch.tensor(data["root_rot"][:, [3, 0, 1, 2]], dtype=torch.float32, device=device)
        self.dof_pos = torch.tensor(data["dof_pos"], dtype=torch.float32, device=device)
        self.num_frames = self.dof_pos.shape[0]

    def _load_qpos_npz(self, motion_file: str, device: torch.device):
        data = np.load(motion_file, allow_pickle=True)
        self.fps = float(data["fps"][0])
        self.dof_names = [str(n) for n in data["robot_joint_names"]]
        qpos = data["qpos"].astype(np.float32)
        self.root_pos = torch.tensor(qpos[:, :3], dtype=torch.float32, device=device)
        # The qpos npz follows the MuJoCo convention, i.e. quaternions are already wxyz.
        self.root_quat = torch.tensor(qpos[:, 3:7], dtype=torch.float32, device=device)
        self.dof_pos = torch.tensor(qpos[:, 7:], dtype=torch.float32, device=device)
        self.num_frames = self.dof_pos.shape[0]

    def _resample(self, device: torch.device):
        times_in = torch.arange(self.num_frames, dtype=torch.float32, device=device) / self.fps
        duration = times_in[-1]
        times_out = torch.arange(0.0, duration, 1.0 / TRACKING_FPS, dtype=torch.float32, device=device)
        phase = times_out * self.fps
        idx0 = torch.floor(phase).long()
        idx1 = torch.clamp(idx0 + 1, max=self.num_frames - 1)
        blend = (phase - idx0).unsqueeze(-1)

        self.root_pos = (1.0 - blend) * self.root_pos[idx0] + blend * self.root_pos[idx1]
        self.dof_pos = (1.0 - blend) * self.dof_pos[idx0] + blend * self.dof_pos[idx1]
        self.root_quat = _slerp(self.root_quat[idx0], self.root_quat[idx1], blend)
        self.num_frames = times_out.shape[0]
        print(f"[INFO]: Resampled {self.fps:.0f} Hz -> {TRACKING_FPS:.0f} Hz, {self.num_frames} frames")

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion: MotionLoader, output_file: str):
    robot: Articulation = scene["robot"]
    robot_joint_indexes = robot.find_joints(motion.dof_names, preserve_order=True)[0]

    log = {
        "fps": [TRACKING_FPS],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    num_frames = motion.num_frames if args_cli.max_frames <= 0 else min(motion.num_frames, args_cli.max_frames)

    for i in range(num_frames):
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.root_pos[i] + scene.env_origins
        root_states[:, 3:7] = motion.root_quat[i]
        root_states[:, 7:10] = motion.root_lin_vel[i]
        root_states[:, 10:] = motion.root_ang_vel[i]
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion.dof_pos[i]
        joint_vel[:, robot_joint_indexes] = motion.dof_vel[i]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()  # No physics: we only want forward kinematics.
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[key] = np.stack(log[key], axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    np.savez(output_file, **log)
    print(f"[INFO]: Saved {num_frames} frames to {output_file}")


def _collect_inputs(input_path: str, output_path: str) -> list[tuple[str, str]]:
    """Expand the CLI input into a list of (input_file, output_file) pairs."""
    if os.path.isdir(input_path):
        if output_path.lower().endswith(".npz"):
            raise ValueError("When --input_file is a directory, --output_file must be a directory as well.")
        out_dir = os.path.abspath(output_path)
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".pkl", ".npz"))
        )
        files = [f for f in files if os.path.dirname(os.path.abspath(f)) != out_dir]
        if not files:
            raise ValueError(f"No .pkl/.npz files found in {input_path}.")
        return [(f, os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".npz")) for f in files]
    return [(input_path, output_path)]


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / TRACKING_FPS
    sim = SimulationContext(sim_cfg)

    scene = InteractiveScene(ConvertSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    jobs = _collect_inputs(args_cli.input_file, args_cli.output_file)
    for input_file, output_file in jobs:
        print(f"[INFO]: Converting {input_file} -> {output_file}")
        motion = MotionLoader(input_file, sim.device)
        run_simulator(sim, scene, motion, output_file)
    print(f"[INFO]: Converted {len(jobs)} motion(s).")


if __name__ == "__main__":
    main()
    simulation_app.close()
