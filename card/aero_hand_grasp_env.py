# aero_hand_grasp_env.py
#
# Grasp-and-hold task for the TetherIA Aero Hand Open, ported to Isaac Lab.
#
# CHANGE (16 -> 7 action dims): the real hand only has 7 motors driving 16 joints
# through tendons, and that's what TetherIA's own SDK and mujoco_playground RL env
# control/observe in ("tendon space"), not 16 independent joints. See:
#   https://docs.tetheria.ai/docs/sdk/#joints-to-actuations-mapping
#   https://docs.tetheria.ai/docs/sdk/#compact-joint-representation
#   https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/aero_hand/rotate_z.py
#
# Isaac Lab/PhysX doesn't model spatial tendons the way MuJoCo does, so this env
# reproduces TetherIA's "compact representation" instead: each of the 7 actuator
# commands is broadcast identically to every joint it drives (ACTUATOR_TO_JOINTS
# below). That's an exact match for our case because, per joint_limits_deg, every
# joint inside a group shares the same (lower, upper) range.
from __future__ import annotations
import math

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from aero_hand_cfg import AERO_HAND_CFG
from card_cfg import CARD_CFG

# ---------------------------------------------------------------------------
# 7-tendon-actuator <-> 16-joint mapping (matches TetherIA's real hardware and
# the "compact representation" table in their SDK docs).
# ---------------------------------------------------------------------------
ACTUATOR_NAMES = [
    "right_thumb_cmc_abd_act",
    "right_thumb_cmc_flex_act",
    "right_thumb_tendon_act",   # drives right_thumb_mcp + right_thumb_ip together
    "right_index_tendon_act",   # drives right_index_mcp_flex + pip + dip together
    "right_middle_tendon_act",  # drives right_middle_mcp_flex + pip + dip together
    "right_ring_tendon_act",    # drives right_ring_mcp_flex + pip + dip together
    "right_pinky_tendon_act",   # drives right_pinky_mcp_flex + pip + dip together
]

ACTUATOR_TO_JOINTS = {
    "right_thumb_cmc_abd_act":  ["right_thumb_cmc_abd"],
    "right_thumb_cmc_flex_act": ["right_thumb_cmc_flex"],
    "right_thumb_tendon_act":   ["right_thumb_mcp", "right_thumb_ip"],
    "right_index_tendon_act":   ["right_index_mcp_flex", "right_index_pip", "right_index_dip"],
    "right_middle_tendon_act":  ["right_middle_mcp_flex", "right_middle_pip", "right_middle_dip"],
    "right_ring_tendon_act":    ["right_ring_mcp_flex", "right_ring_pip", "right_ring_dip"],
    "right_pinky_tendon_act":   ["right_pinky_mcp_flex", "right_pinky_pip", "right_pinky_dip"],
}


@configclass
class AeroHandGraspEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 7                # one normalized [-1, 1] target per tendon actuator
    observation_space = 68          # 16 pos + 16 vel + 16 torque + 7 prev-action + 3 card pos
                                     # + 3 card linvel + 3 card angvel + 4 card quat (see _get_observations)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # assets
    hand_cfg: ArticulationCfg = AERO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    card_cfg: RigidObjectCfg = CARD_CFG.replace(prim_path="/World/envs/env_.*/Card")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=0.6, replicate_physics=True
    )

    # joint limits (degrees), from the TetherIA SDK docs (https://docs.tetheria.ai/docs/sdk/)
    joint_limits_deg = {
        "right_index_mcp_flex":  (0.0, 90.000206),
        "right_index_pip":       (0.0, 90.000206),
        "right_index_dip":       (0.0, 90.000206),
        "right_middle_mcp_flex": (0.0, 90.000206),
        "right_middle_pip":      (0.0, 90.000206),
        "right_middle_dip":      (0.0, 90.000206),
        "right_ring_mcp_flex":   (0.0, 90.000206),
        "right_ring_pip":        (0.0, 90.000206),
        "right_ring_dip":        (0.0, 90.000206),
        "right_pinky_mcp_flex":  (0.0, 90.000206),
        "right_pinky_pip":       (0.0, 90.000206),
        "right_pinky_dip":       (0.0, 90.000206),
        "right_thumb_cmc_abd":   (0.0, 99.99833),
        "right_thumb_cmc_flex":  (0.0, 54.76903),
        "right_thumb_mcp":       (0.0, 90.000206),
        "right_thumb_ip":        (0.0, 90.000206),
    }

    # grasp / reset — full 16-joint tuned pose, UNCHANGED from before. This is written
    # directly into the sim as the reset state via write_joint_state_to_sim(), so it is
    # NOT limited by the 7-actuator coupling: every joint still gets its own hand-tuned
    # value at reset time. The 7-dim tendon-space equivalent used for control/rewards
    # (self._grasp_actuation_pos) is derived from this automatically in __init__.
    grasp_joint_pos = {
        "right_thumb_cmc_abd":  1.25, # base of thumb
        "right_thumb_cmc_flex": 0.1, # moves thumb across the palm
        "right_thumb_mcp":      0.7, # moves thumb up/down
        "right_thumb_ip":       1.0, # tip of thumb

        "right_index_mcp_flex": 1.6,
        "right_index_pip":      0.55,
        "right_index_dip":      0.6,

        "right_middle_mcp_flex": 0.0,
        "right_middle_pip":      0.0,
        "right_middle_dip":      0.0,

        "right_ring_mcp_flex": 0.0,
        "right_ring_pip":      0.0,
        "right_ring_dip":      0.0,

        "right_pinky_mcp_flex": 0.0,
        "right_pinky_pip":      0.0,
        "right_pinky_dip":      0.0,
    }

    card_drop_height_thresh = 0.15    # meters of downward drift counted as "dropped"

    # reward scales — named/structured after TetherIA's own aero_hand reward terms
    # (angvel/linvel/pose/torques/energy/action_rate/termination) in rotate_z.py,
    # adapted from "rotate as fast as possible" to "hold still without dropping".
    rew_scale_card_height = 5.0      # reward each step the card is still considered "held"
    rew_scale_card_still = -0.1      # penalize card linear speed -> encourage a stable, non-wobbly grasp
    rew_scale_action_rate = -0.01   # penalize jerky tendon commands (act - last_act)^2
    rew_scale_pose = -0.05          # pull the (16-joint) hand shape back toward grasp_joint_pos
    rew_scale_torques = -1.0e-3     # penalize actuator torque / cable tension — PLACEHOLDER, tune
    rew_scale_energy = -1.0e-3      # penalize |vel|*|torque| wasted mechanical power — PLACEHOLDER, tune
    rew_scale_drop_penalty = -10.0


class AeroHandGraspEnv(DirectRLEnv):
    cfg: AeroHandGraspEnvCfg

    def __init__(self, cfg: AeroHandGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, self._joint_names = self.hand.find_joints("right_.*")
        joint_index = {name: i for i, name in enumerate(self._joint_names)}

        lower_deg = [self.cfg.joint_limits_deg[name][0] for name in self._joint_names]
        upper_deg = [self.cfg.joint_limits_deg[name][1] for name in self._joint_names]
        self._joint_lower = torch.tensor([math.radians(d) for d in lower_deg], device=self.device)
        self._joint_upper = torch.tensor([math.radians(d) for d in upper_deg], device=self.device)

        grasp_values = [self.cfg.grasp_joint_pos[name] for name in self._joint_names]
        self._grasp_joint_pos = torch.tensor(grasp_values, device=self.device)  # (16,)

        # --- Build the 7-actuator <-> 16-joint mapping ---
        num_joints = len(self._joint_names)
        num_act = len(ACTUATOR_NAMES)

        act_to_joint_matrix = torch.zeros(num_joints, num_act, device=self.device)
        act_lower_deg, act_upper_deg, grasp_actuation_pos = [], [], []
        for a_idx, act_name in enumerate(ACTUATOR_NAMES):
            joints = ACTUATOR_TO_JOINTS[act_name]
            for jname in joints:
                act_to_joint_matrix[joint_index[jname], a_idx] = 1.0
            # every joint inside a group shares the same (lower, upper) range in the
            # SDK's table, so it's safe to just read it off the first one.
            lo, hi = self.cfg.joint_limits_deg[joints[0]]
            act_lower_deg.append(lo)
            act_upper_deg.append(hi)
            # collapse the tuned per-joint grasp pose into a single actuator value by
            # averaging across the joints that actuator drives together (a real tendon
            # can't hold thumb_mcp and thumb_ip at two different angles independently).
            grasp_actuation_pos.append(sum(self.cfg.grasp_joint_pos[j] for j in joints) / len(joints))

        self._act_to_joint_matrix = act_to_joint_matrix                                      # (16, 7)
        self._act_lower = torch.tensor([math.radians(d) for d in act_lower_deg], device=self.device)  # (7,)
        self._act_upper = torch.tensor([math.radians(d) for d in act_upper_deg], device=self.device)  # (7,)
        self._grasp_actuation_pos = torch.tensor(grasp_actuation_pos, device=self.device)    # (7,)

        self._last_action = torch.zeros(self.num_envs, num_act, device=self.device)

        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel

        # per-env card height captured at reset time, used to detect "has it dropped"
        self._initial_card_height = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.hand_cfg)
        self.card = RigidObject(self.cfg.card_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["card"] = self.card

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: (num_envs, 7) -- one normalized [-1, 1] command per tendon actuator.
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        # map each actuator's normalized command to its own radian range, then broadcast
        # it out to every joint that actuator drives (ACTUATOR_TO_JOINTS / TetherIA's
        # "compact representation" — see module docstring at the top of this file).
        act_targets = self._act_lower + (self.actions + 1.0) * 0.5 * (self._act_upper - self._act_lower)
        joint_targets = act_targets @ self._act_to_joint_matrix.T   # (num_envs, 7) -> (num_envs, 16)
        self.hand.set_joint_position_target(joint_targets, joint_ids=self._joint_ids)

        # persisted so external scripts (e.g. the ROS2 sim-to-real bridge) can read
        # exactly what was commanded this step -- shape (num_envs, 16), columns in
        # self._joint_names order. This is the *commanded* target, not the achieved
        # joint_pos -- deliberately, so what reaches real hardware matches what the
        # policy/action actually asked for, same as it would from a real controller.
        self.joint_targets = joint_targets

        # debug print of the actual 7-dof control signal for the first env only.
        # Compare against joint_pos in _get_observations()'s debug print: joint_targets
        # below are hard duplicates within each actuator's group, every step, no
        # exceptions -- that's the real 7-dof constraint, enforced here. joint_pos is
        # the *achieved* physics state (16 independently PD-controlled joints) and can
        # legitimately drift apart from these targets, and from each other -- see the
        # explanation in chat for why. NOTE: _apply_action runs once per physics
        # substep (decimation=2), so this prints twice per env.step(), not once.
        # Remove before training -- .item()/.tolist() force a GPU->CPU sync every call.
        if joint_targets.shape[0] > 0:
            env0 = 0
            print("=" * 70)
            print("_apply_action() debug for env 0 -- this IS the 7-dof control signal")
            print("actions (raw, normalized [-1,1]):", [f"{v:.3f}" for v in self.actions[env0].tolist()])
            print("act_targets (radians, one per actuator):")
            for name, val in zip(ACTUATOR_NAMES, act_targets[env0].tolist()):
                print(f"    {name:28s} {val:.3f}")
            print("joint_targets (radians, broadcast to all 16 -- grouped by actuator):")
            for name in ACTUATOR_NAMES:
                joints = ACTUATOR_TO_JOINTS[name]
                vals = {j: f"{joint_targets[env0, self._joint_names.index(j)].item():.3f}" for j in joints}
                print(f"    {name:28s} {vals}")
            print("=" * 70)

    def _get_observations(self) -> dict:
        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel

        joint_pos = self.joint_pos[:, self._joint_ids]
        joint_vel = self.joint_vel[:, self._joint_ids]
        # applied_torque is Isaac Lab's post-clipping estimate of what the implicit PD
        # actuator produced this step -- the closest sim analogue of the real hand's
        # per-actuator current/torque sensing (get_actuator_currents() in the SDK).
        # NOTE: for implicit actuators this is an *approximation* computed from the PD
        # error (PhysX doesn't expose true per-joint torque for implicit drives) — fine
        # for reward/observation purposes, just don't treat it as a calibrated sensor.
        joint_torque = self.hand.data.applied_torque[:, self._joint_ids]

        # Card pose relative to the hand's articulation root. NOTE: this uses the ROOT,
        # not a "palm" body/site, since I can't confirm the palm body's exact name in
        # your aero_hand.usda. If you want palm- or fingertip-relative distances (closer
        # to what TetherIA's cube_pos_error does), print `self.hand.body_names` once in
        # Isaac Sim and swap in the right body index via self.hand.data.body_pos_w[:, idx].
        hand_root_pos = self.hand.data.root_pos_w
        card_pos_rel = self.card.data.root_pos_w - hand_root_pos
        card_lin_vel = self.card.data.root_lin_vel_w
        card_ang_vel = self.card.data.root_ang_vel_w
        card_quat = self.card.data.root_quat_w

        obs = torch.cat(
            (
                joint_pos,           # 16 - current joint angles
                joint_vel,           # 16 - current joint velocities
                joint_torque,        # 16 - applied actuator torque per joint
                self._last_action,   # 7  - previous normalized tendon-space action
                card_pos_rel,         # 3
                card_lin_vel,         # 3
                card_ang_vel,         # 3
                card_quat,            # 4
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        card_height = self.card.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        card_speed = torch.norm(self.card.data.root_lin_vel_w, dim=-1)

        dropped = (self._initial_card_height - card_height) > self.cfg.card_drop_height_thresh
        held = ~dropped

        # read straight from hand.data (not the cached self.joint_pos/self.joint_vel)
        # so this doesn't depend on whether _get_observations has already run this step.
        joint_pos = self.hand.data.joint_pos[:, self._joint_ids]
        joint_vel = self.hand.data.joint_vel[:, self._joint_ids]
        joint_torque = self.hand.data.applied_torque[:, self._joint_ids]

        rew_held = self.cfg.rew_scale_card_height * held.float()
        rew_still = self.cfg.rew_scale_card_still * card_speed
        rew_drop = self.cfg.rew_scale_drop_penalty * dropped.float()
        rew_action_rate = self.cfg.rew_scale_action_rate * torch.sum(
            torch.square(self.actions - self._last_action), dim=-1
        )
        rew_pose = self.cfg.rew_scale_pose * torch.sum(
            torch.square(joint_pos - self._grasp_joint_pos), dim=-1
        )
        rew_torque = self.cfg.rew_scale_torques * torch.sum(torch.square(joint_torque), dim=-1)
        rew_energy = self.cfg.rew_scale_energy * torch.sum(
            torch.abs(joint_vel) * torch.abs(joint_torque), dim=-1
        )

        self._last_action = self.actions.clone()

        return rew_held + rew_still + rew_drop + rew_action_rate + rew_pose + rew_torque + rew_energy

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        card_height = self.card.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        dropped = (self._initial_card_height - card_height) > self.cfg.card_drop_height_thresh
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # --- Hand: spawn already closed around the card (full 16-joint tuned pose) ---
        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._grasp_joint_pos   # broadcasts (16,) across the batch
        joint_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_hand_root = self.hand.data.default_root_state[env_ids].clone()
        default_hand_root[:, :3] += self.scene.env_origins[env_ids]
        self.hand.write_root_pose_to_sim(default_hand_root[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(default_hand_root[:, 7:], env_ids)

        # --- Card: spawn positioned inside the grasp ---
        default_card_root = self.card.data.default_root_state[env_ids].clone()
        default_card_root[:, :3] += self.scene.env_origins[env_ids]
        self.card.write_root_pose_to_sim(default_card_root[:, :7], env_ids)
        self.card.write_root_velocity_to_sim(default_card_root[:, 7:], env_ids)

        self._initial_card_height[env_ids] = default_card_root[:, 2] - self.scene.env_origins[env_ids, 2]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self._last_action[env_ids] = 0.0