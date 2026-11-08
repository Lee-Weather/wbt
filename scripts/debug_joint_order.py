"""Print the articulation joint/body order of a tracking task.

The MuJoCo replay (scripts/replay_mujoco_policy.py) has to reproduce the exact
Isaac Lab articulation order for the policy's 29-DoF action/observation vectors.
The order for a URDF-spawned articulation is not documented, so print it from the
real environment.

Run on the Isaac Lab platform:

    gm-run wbt/scripts/debug_joint_order.py --task=Tracking-Flat-F1-v0 --headless
"""

import argparse
import sys
from pathlib import Path

# make the extension importable without `pip install -e`
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "whole_body_tracking"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Reference motion npz (default: the F1 B3 walk clip inside the repo).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import whole_body_tracking.tasks  # noqa: F401  (registers the environments)
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
env_cfg.commands.motion.motion_file = args_cli.motion_file or str(
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/assets/f1/motions/converted/B3_walk1_weichai_f1_29.npz"
)
env = gym.make(args_cli.task, cfg=env_cfg)
robot = env.unwrapped.scene["robot"]
print("JOINT_ORDER", list(robot.joint_names), flush=True)
print("BODY_ORDER", list(robot.body_names), flush=True)
print("JOINT_COUNT", len(robot.joint_names), flush=True)
env.close()
simulation_app.close()
