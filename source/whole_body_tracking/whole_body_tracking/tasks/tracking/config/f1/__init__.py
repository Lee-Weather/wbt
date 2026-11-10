import gymnasium as gym

from . import agents, flat_env_cfg, flat_env_cfg_run, flat_env_cfg_walk, velocity_env_cfg

##
# Register Gym environments.
##
# Baseline (1.0× Kp/Kd)
gym.register(
    id="Tracking-Flat-F1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.F1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:F1FlatPPORunnerCfg",
    },
)

# Walking (0.9× Kp/Kd)
gym.register(
    id="Tracking-Flat-F1-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg_walk.F1FlatWalkEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:F1FlatPPORunnerCfg",
    },
)

# Running (1.1× Kp/Kd)
gym.register(
    id="Tracking-Flat-F1-Run-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg_run.F1FlatRunEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:F1FlatPPORunnerCfg",
    },
)

# Velocity-conditioned locomotion (command-driven, with reference prior)
gym.register(
    id="Tracking-Flat-F1-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": velocity_env_cfg.F1VelocityEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:F1FlatPPORunnerCfg",
    },
)
