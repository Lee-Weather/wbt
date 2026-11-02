"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import glob
import shutil
import sys
import time
from pathlib import Path

# make the extension importable without `pip install -e` (e.g. on the training platform)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "whole_body_tracking"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--checkpoint_url",
    type=str,
    default=None,
    help="Signed URL of a model .pt file. Downloaded at runtime when the run is not available locally.",
)
parser.add_argument(
    "--checkpoint_url_b64",
    type=str,
    default=None,
    help="URL-safe Base64 of a signed model URL. Preferred on the cloud platform (avoids '&' splitting startScript).",
)
parser.add_argument("--num_steps", type=int, default=0, help="Stop after this many policy steps (0 = run forever).")
parser.add_argument(
    "--checkpoint_name",
    type=str,
    default=None,
    help="Local filename for the downloaded checkpoint (use model_<iteration>.pt so the platform registers it).",
)
parser.add_argument(
    "--no_termination",
    action="store_true",
    help="Disable early terminations (useful for recording a full-length video).",
)
parser.add_argument(
    "--dump_obs",
    type=int,
    default=0,
    help="Dump the first N steps of (obs, action) to gm_play/obs_dump.csv packaged as a .pt artifact.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint_url_b64:
        import base64

        args_cli.checkpoint_url = base64.urlsafe_b64decode(args_cli.checkpoint_url_b64.encode()).decode()

    if args_cli.checkpoint_url:
        import urllib.request

        temp_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, "gm_play"))
        os.makedirs(temp_dir, exist_ok=True)
        resume_path = os.path.join(temp_dir, args_cli.checkpoint_name or "model.pt")
        print(f"[INFO]: Downloading checkpoint from URL to {resume_path}", flush=True)
        urllib.request.urlretrieve(args_cli.checkpoint_url, resume_path)
        print(f"[INFO]: Download finished ({os.path.getsize(resume_path)} bytes)", flush=True)
    elif args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}", flush=True)
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    if args_cli.no_termination:
        env_cfg.terminations.anchor_pos = None
        env_cfg.terminations.anchor_ori = None
        env_cfg.terminations.ee_body_pos = None
        print("[INFO]: Early terminations disabled", flush=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "play_output",
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    print(f"[INFO]: Exported ONNX to {export_model_dir}/policy.onnx", flush=True)
    # reset environment
    obs, _ = env.get_observations()
    print("[INFO]: Environment ready, starting rollout", flush=True)
    timestep = 0
    done_count = 0
    diag_rows = []
    obs_rows = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if args_cli.dump_obs > 0 and timestep < args_cli.dump_obs:
                obs_vec = obs["policy"][0] if isinstance(obs, dict) else obs[0]
                row = [timestep] + obs_vec.detach().cpu().tolist() + actions[0].detach().cpu().tolist()
                obs_rows.append(row)
            # env stepping
            obs, _, dones, _ = env.step(actions)
        timestep += 1
        done_count += int(dones.sum().item())
        episode_reset = int(dones[0].item())
        term_mgr = env.unwrapped.termination_manager
        step_terminated = int(term_mgr.terminated[0].item())
        step_time_out = int(term_mgr.time_outs[0].item())
        cmd_term = env.unwrapped.command_manager.get_term("motion")
        robot = env.unwrapped.scene["robot"]
        diag_rows.append([
            timestep,
            robot.data.root_pos_w[0, 0].item(),
            robot.data.root_pos_w[0, 1].item(),
            robot.data.root_pos_w[0, 2].item(),
            cmd_term.metrics["error_anchor_pos"][0].item(),
            cmd_term.metrics["error_anchor_rot"][0].item(),
            cmd_term.metrics["error_body_pos"][0].item(),
            cmd_term.metrics["error_body_rot"][0].item(),
            actions.abs().mean().item(),
            episode_reset,
            step_terminated,
            step_time_out,
        ])
        if timestep % 50 == 0:
            base_z = robot.data.root_pos_w[:, 2]
            print(
                f"[INFO]: step={timestep} base_z_mean={base_z.mean().item():.3f} "
                f"done={done_count} |a|={actions.abs().mean().item():.3f} "
                f"anchor_err={cmd_term.metrics['error_anchor_pos'][0].item():.3f} "
                f"body_err={cmd_term.metrics['error_body_pos'][0].item():.3f}",
                flush=True,
            )
        if args_cli.num_steps > 0 and timestep >= args_cli.num_steps:
            break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    print(f"[INFO]: Rollout finished: steps={timestep}, done={done_count}", flush=True)

    # diagnostics CSV + PT package for the platform artifact pipeline
    diag_dir = os.path.join(log_root_path, "gm_play")
    os.makedirs(diag_dir, exist_ok=True)
    csv_path = os.path.join(diag_dir, "isaac_diag.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "base_x", "base_y", "base_z",
            "error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot", "action_abs_mean",
            "episode_reset", "step_terminated", "step_time_out",
        ])
        writer.writerows(diag_rows)
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    pt_path = os.path.join(diag_dir, "model_isaac_csv.pt")
    torch.save({"bytes": csv_bytes}, pt_path)
    print(f"[INFO]: Wrote {csv_path} ({len(diag_rows)} rows) and {pt_path}", flush=True)

    if obs_rows:
        n_obs = len(obs_rows[0]) - 1 - 29
        obs_csv_path = os.path.join(diag_dir, "obs_dump.csv")
        with open(obs_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"] + [f"obs_{i}" for i in range(n_obs)] + [f"act_{i}" for i in range(29)]
            writer.writerow(header)
            writer.writerows(obs_rows)
        with open(obs_csv_path, "rb") as f:
            obs_bytes = f.read()
        obs_pt_path = os.path.join(diag_dir, "model_obs_dump.pt")
        torch.save({"bytes": obs_bytes}, obs_pt_path)
        print(f"[INFO]: Wrote {obs_csv_path} ({len(obs_rows)} rows, {n_obs} obs dims) and {obs_pt_path}", flush=True)

    # close the simulator (flushes the recorded video)
    env.close()

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "play")
        videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        if videos:
            target = os.path.join(log_root_path, "play_output.mp4")
            shutil.copyfile(videos[-1], target)
            target_gm = os.path.join(log_root_path, "gm_play", "play_output.mp4")
            shutil.copyfile(videos[-1], target_gm)
            print(f"[INFO]: Video saved to {target} and {target_gm} ({os.path.getsize(target)} bytes)", flush=True)
        else:
            print(f"[WARN]: No mp4 found in {video_dir}", flush=True)

    # Give the platform SDK time to finish scanning/uploading the generated artifacts.
    print("[INFO]: Holding 60s for artifact upload", flush=True)
    time.sleep(60)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
