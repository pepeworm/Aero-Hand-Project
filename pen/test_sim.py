# test.py
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from aero_hand_grasp_env import AeroHandGraspEnv, AeroHandGraspEnvCfg

cfg = AeroHandGraspEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
env = AeroHandGraspEnv(cfg)

def grasp_hold_actions(env):
    # 7-dim tendon-actuator equivalents of the old 16-dim _joint_lower/_joint_upper/
    # _grasp_joint_pos, built in __init__ by collapsing the tuned per-joint grasp pose
    # down to what the 7 real tendon actuators can actually command.
    lower = env._act_lower
    upper = env._act_upper
    grasp = env._grasp_actuation_pos
    normalized = 2.0 * (grasp - lower) / (upper - lower) - 1.0
    return normalized.unsqueeze(0).repeat(env.num_envs, 1)

hold_actions = grasp_hold_actions(env)

obs, _ = env.reset()
while simulation_app.is_running():
    # actions = torch.zeros(env.num_envs, 7, device=env.device)   # hold the grasp pose
    # obs, rew, terminated, truncated, info = env.step(actions)
    
    obs, rew, terminated, truncated, info = env.step(hold_actions)

env.close()
simulation_app.close()


# test.py
# import argparse
# from isaaclab.app import AppLauncher

# parser = argparse.ArgumentParser()
# parser.add_argument("--num_envs", type=int, default=9)   # small number = fast reload while tuning
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import torch
# from aero_hand_grasp_env import AeroHandGraspEnv, AeroHandGraspEnvCfg

# cfg = AeroHandGraspEnvCfg()
# cfg.scene.num_envs = args_cli.num_envs
# env = AeroHandGraspEnv(cfg)


# def grasp_hold_actions(env):
#     lower = env._act_lower
#     upper = env._act_upper
#     grasp = env._grasp_actuation_pos
#     normalized = 2.0 * (grasp - lower) / (upper - lower) - 1.0
#     return normalized.unsqueeze(0).repeat(env.num_envs, 1)


# # --- Reset: this is where your per-joint grasp_joint_pos gets applied ---
# obs, info = env.reset()
# print("=" * 60)
# print("RESET")
# print("=" * 60)
# print(f"obs['policy'] shape: {obs['policy'].shape}")   # (num_envs, 68)
# print(f"obs['policy'][0]:    {obs['policy'][0]}")
# print(f"info: {info}")

# hold_actions = grasp_hold_actions(env)


# # --- Step a few times, print exactly what a training loop would get ---
# # for step in range(5):
# #     obs, rew, terminated, truncated, info = env.step(hold_actions)
# #     print("-" * 60)
# #     print(f"STEP {step}")
# #     print(f"  reward[:5]:        {rew[:5]}")
# #     print(f"  terminated.sum():  {terminated.sum().item()}  (envs that dropped the mug)")
# #     print(f"  truncated.sum():   {truncated.sum().item()}   (envs that hit episode length)")
# step = 0

# # --- Keep running so you can watch the grip in the viewport ---
# print("=" * 60)
# print("Holding grasp pose. Watch the viewport to check the grip.")
# print("=" * 60)
# while simulation_app.is_running():
#     obs, rew, terminated, truncated, info = env.step(hold_actions)
#     step += 1

#     print("-" * 60)
#     print(f"STEP {step}")
#     print(f"  reward[:5]:        {rew[:5]}")
#     print(f"  terminated.sum():  {terminated.sum().item()}  (envs that dropped the mug)")
#     print(f"  truncated.sum():   {truncated.sum().item()}   (envs that hit episode length)")

# env.close()
# simulation_app.close()

# obs — a dict, not a bare tensor. Policy network reads obs["policy"], shape (num_envs, 68).
# rew — tensor, shape (num_envs,), one scalar reward per environment.
# terminated — bool tensor, shape (num_envs,). True where the mug dropped past mug_drop_height_thresh.
# truncated — bool tensor, shape (num_envs,). True where episode_length_s elapsed.
# info — dict, empty by default unless you add extra logging.
