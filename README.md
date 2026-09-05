# TetherIA Aero Hand Open: Grasp-and-Hold (Isaac Lab)

Isaac Lab port of the TetherIA Aero Hand Open for a grasp-and-hold task: the hand is
fixed in the air above the ground plane, rotated to face downward, spawns already
closed around an object, and has to hold that object against gravity for the length
of the episode without dropping it.

The codebase is instantiated once per graspable object (currently **mug**,
**bottle**, **card**, and **pen**) as four sibling directories: `mug/`, `bottle/`,
`card/`, `pen/`. Each directory is a self-contained copy of the same five files, and
only the object config and the tuned grasp pose actually differ between them. See
[Per-object reference](#per-object-reference) for the exact numbers.

The action space, observation space, and reward *structure* mirror TetherIA's own
official RL environment in `mujoco_playground`. (Their published env is a
cube-rotation task, not grasp-hold, and they have no grasp task released. The core
conventions, tendon-space control and torque/energy-aware rewards, carry over
regardless of task or object.)

There is also a **hardware path**, kept entirely in `real_hand/`:
`real_hand/aero_hand_bridge.py` drives the physical hand over ROS2 using the same
7-actuator convention and ordering the sim policy outputs. Nothing in `real_hand/`
imports Isaac Lab, and nothing in the scene variants imports ROS2, so the split is
clean in both directions. See [Running on the real hand](#running-on-the-real-hand).

---

## Repository layout

```
hand_isaaclab/
├── mug/ bottle/ card/ pen/     one directory per graspable object (see below)
├── hand_model/                 USD assets
│   ├── aero_hand.usda          the 16-joint Aero Hand Open (right hand)
│   ├── only_cup.usda           mug
│   ├── only_bottle.usda        bottle
│   ├── only_credit_card.usda   card
│   └── only_pen.usda           pen
├── real_hand/                  physical hand over ROS2 (no Isaac Lab imports)
│   ├── aero_hand_bridge.py
│   └── test_ros.py
├── mounts/                     printable hardware for the physical rig
│   ├── piperx_adapter.stl      wrist adapter for the AgileX PiPER arm
│   ├── table_clamp.stl
│   └── table_clamp_screw.stl
├── website/                    static documentation site (index.html + public/)
└── OLD/                        stale bytecode from a previous layout; not tracked
```

Every object directory contains exactly these five files:

| File | What it is |
|---|---|
| `aero_hand_cfg.py` | Hand asset: USD path, base pose/orientation, per-joint actuator gains. **Byte-identical across all four directories.** |
| `aero_hand_scene_cfg.py` | An `InteractiveSceneCfg` with a ground plane, dome light, and N cloned hands. **Byte-identical across all four, and currently unused**: nothing imports it. The env builds its own scene in `_setup_scene()`. Kept as a standalone hand-only scene definition. |
| `<object>_cfg.py` | The graspable object: USD path, mass, spawn position and rotation. `mug_cfg.py`, `bottle_cfg.py`, `card_cfg.py`, `pen_cfg.py`. |
| `aero_hand_grasp_env.py` | The `DirectRLEnv` and its `DirectRLEnvCfg`: action space, observation space, rewards, termination, reset. Same structure everywhere, with the object's name substituted into the object-specific fields. |
| `test_sim.py` | Standalone viewer: spawns N envs and holds the tuned grasp pose so you can eyeball the grip. No RL loop. **Byte-identical across all four directories.** It imports `aero_hand_grasp_env` from its own directory, so which object it loads is decided by which directory you run it from. |

The env class and cfg class are named `AeroHandGraspEnv` / `AeroHandGraspEnvCfg` in
all four directories. They are *not* importable side by side in one process, because
the module names collide.

---

## Requirements

- **Isaac Sim + Isaac Lab** (the sim path). Scripts are launched through Isaac Lab's
  Python, either `./isaaclab.sh -p <script>` from your Isaac Lab install or
  `python <script>` from inside an activated Isaac Lab conda env.
- **PyTorch**, which comes with Isaac Lab.
- **ROS2 Humble**, the `aero-hand-open` workspace built with `colcon`, and the
  `aero_open_sdk` Python package (the hardware path only; see
  [Running on the real hand](#running-on-the-real-hand)).

The two paths are independent: you can run the sim with no ROS2 installed, and run
the hardware with no Isaac Lab installed.

---

## Running it

### Checking a tuned grasp pose

```bash
cd mug            # or bottle / card / pen
python test_sim.py                 # 100 envs (default)
python test_sim.py --num_envs 9    # fewer envs = faster reload while tuning
```

> **You must `cd` into the object directory first.** Both `aero_hand_cfg.py` and
> `<object>_cfg.py` reference their USD assets with paths relative to the working
> directory (`../hand_model/aero_hand.usda`), so running `python mug/test_sim.py`
> from the repo root fails to resolve the assets.

`test_sim.py` builds the real `AeroHandGraspEnv` (object included, actions pushed
through the actual 7-tendon mapping), then computes the normalized 7-dim action that
corresponds to the variant's tuned grasp pose and steps forever with it:

```python
lower, upper, grasp = env._act_lower, env._act_upper, env._grasp_actuation_pos
normalized = 2.0 * (grasp - lower) / (upper - lower) - 1.0
```

So what you see in the viewport is the pose the 7 actuators can actually *hold*, not
the 16-joint reset pose. Those differ whenever a group's joints were tuned to
different angles. Use it to check whether the grip survives contact and gravity.

`--num_envs` is the only argument the script adds; everything else
(`--headless`, `--device`, …) comes from Isaac Lab's `AppLauncher`.

### Training

Every variant is an ordinary Isaac Lab `DirectRLEnv` with `action_space = 7`,
`observation_space = 68`, and `state_space = 0`, so it drops into whatever
PPO/training harness you already use. There is no training script, task
registration, or `gym.register` call in this repo; you instantiate
`AeroHandGraspEnvCfg` and `AeroHandGraspEnv` directly, the same way `test_sim.py`
does.

**Before you train, remove the debug print block in `_apply_action()`** (see
[Known placeholders](#known-placeholders-check-before-training)). It dumps the full
control signal for env 0 on every physics substep and forces a GPU→CPU sync each
call.

### The real hand

```bash
ros2 run aero_hand_open aero_hand_node --ros-args \
    -p right_port:=auto -p control_space:=joint
python3 real_hand/test_ros.py
```

Full setup, SDK patch, and network notes are in
[Running on the real hand](#running-on-the-real-hand).

---

## The simulation environment

### `AeroHandGraspEnvCfg` at a glance

| Field | Value | Note |
|---|---|---|
| `decimation` | `2` | physics substeps per policy step |
| `episode_length_s` | `10.0` | 600 policy steps per episode |
| `action_space` | `7` | one normalized `[-1, 1]` target per tendon actuator |
| `observation_space` | `68` | see [Observations](#observations-68-dim) |
| `state_space` | `0` | no asymmetric critic observation |
| `sim.dt` | `1/120` | **120 Hz physics → 60 Hz policy** with `decimation = 2` |
| `sim.render_interval` | `decimation` | render once per policy step |
| `scene.num_envs` | `100` | overridden by `test_sim.py --num_envs` |
| `scene.env_spacing` | `0.6` | metres between cloned envs |
| `scene.replicate_physics` | `True` | required for the GPU cloner fast path |
| `joint_limits_deg` | 16 entries | from the TetherIA SDK docs |
| `grasp_joint_pos` | 16 entries | per-object tuned reset pose, radians |
| `<object>_drop_height_thresh` | `0.15` | metres of downward drift counted as dropped |

### Hand asset: `aero_hand_cfg.py`

```python
AERO_HAND_USD_PATH = "../hand_model/aero_hand.usda"
```

`ArticulationCfg` with:

- **Rigid body props**: gravity enabled, `max_depenetration_velocity = 5.0`.
- **Articulation root props**: `enabled_self_collisions = True` (fingers can touch
  the palm and each other), 8 position-solver iterations, 0 velocity-solver
  iterations, and `fix_root_link = True`. That last one means **the hand is bolted
  in place**: it cannot translate or rotate, only its 16 joints move.
- **Initial state**: `pos = (0.0, 0.0, 1.0)` (1 m above the ground plane) and
  `rot = (0.173648, 0.0, 0.984807, 0.0)` in Isaac Lab's `(w, x, y, z)` order. That
  quaternion is a **160° rotation about +Y**, 20° short of a full 180° flip, which
  points the hand downward while tilting it back toward the camera. (The inline
  comment in the file describes it as 20° about +X. The axis in the comment does not
  match the quaternion, and the quaternion is what the sim uses.) All 16 joints
  start at `0.0` here; the grasp pose is written over this at reset by the env.
- **Actuators**: a single `ImplicitActuatorCfg` matching `right_.*`, i.e. all 16
  joints share one gain set: `stiffness = 15.0`, `damping = 1.0`,
  `effort_limit_sim = 2.0`, `velocity_limit_sim = 5.0`. Stiffness and effort limit
  are placeholders and have not been matched to the real servos.

Because the orientation is a raw quaternion rather than an Euler triple, nudging it
by eye means editing four numbers at once. If you tune this often, converting it back
to a `(roll, pitch, yaw)` degrees constant is the easier workflow.

### Object asset: `<object>_cfg.py`

A single `RigidObjectCfg` per file: USD path, `RigidBodyPropertiesCfg`
(gravity on, `max_depenetration_velocity = 1.0`), `MassPropertiesCfg`, and an
`InitialStateCfg` giving the spawn position in world coordinates (**not** relative to
the hand; the hand root is at `z = 1.0`, so the object's `z` is what places it
inside the grip) plus an optional spawn rotation.

Each of these files also carries a **commented-out "for testing" variant** that
disables gravity and collision, so you can drop the object into the grip and inspect
finger placement without physics fighting you. Swapping between them means
commenting one block and uncommenting the other. `pen_cfg.py` currently has the
**testing block active**: the pen has no gravity and no collision as checked in.

### The 7-tendon action space

The real hand has 7 motors driving 16 joints through tendons, so it is
under-actuated and groups of joints are mechanically forced to move together. This is
a property of the *hand*, not the object, so it is identical across every variant.
The policy outputs 7 numbers (one per actuator, normalized to `[-1, 1]`), and each is
broadcast to every joint it drives:

| Actuator (`ACTUATOR_NAMES`) | Joints it drives (`ACTUATOR_TO_JOINTS`) |
|---|---|
| `right_thumb_cmc_abd_act` | `right_thumb_cmc_abd` |
| `right_thumb_cmc_flex_act` | `right_thumb_cmc_flex` |
| `right_thumb_tendon_act` | `right_thumb_mcp`, `right_thumb_ip` |
| `right_index_tendon_act` | `right_index_mcp_flex`, `right_index_pip`, `right_index_dip` |
| `right_middle_tendon_act` | `right_middle_mcp_flex`, `right_middle_pip`, `right_middle_dip` |
| `right_ring_tendon_act` | `right_ring_mcp_flex`, `right_ring_pip`, `right_ring_dip` |
| `right_pinky_tendon_act` | `right_pinky_mcp_flex`, `right_pinky_pip`, `right_pinky_dip` |

This is TetherIA's "compact representation" from their SDK docs
(`docs.tetheria.ai/docs/sdk/#compact-joint-representation`), reimplemented in Isaac
Lab as a broadcast matrix because PhysX does not model MuJoCo-style spatial tendons.

**How it is built** (`AeroHandGraspEnv.__init__`):

1. `self._act_to_joint_matrix`: a `(16, 7)` matrix of ones and zeros, one column per
   actuator, marking every joint that actuator drives.
2. `self._act_lower` / `self._act_upper`: `(7,)` radian ranges, read off the *first*
   joint in each group. Safe because every joint inside a group shares the same
   `(lower, upper)` in the SDK's table.
3. `self._grasp_actuation_pos`: `(7,)`, the tuned 16-joint grasp pose collapsed by
   **averaging** across each group. A real tendon cannot hold `thumb_mcp` and
   `thumb_ip` at two different angles independently.

**How it is applied** (`_apply_action`, once per physics substep):

```python
act_targets   = self._act_lower + (self.actions + 1.0) * 0.5 * (self._act_upper - self._act_lower)
joint_targets = act_targets @ self._act_to_joint_matrix.T      # (N, 7) -> (N, 16)
self.hand.set_joint_position_target(joint_targets, joint_ids=self._joint_ids)
self.joint_targets = joint_targets     # persisted for external readers
```

`self.joint_targets` is kept on the env deliberately, so an external script (a
sim-to-real bridge, a logger) can read exactly what was *commanded* this step:
`(num_envs, 16)`, columns in `self._joint_names` order. It is the command, not the
achieved `joint_pos`, which is what a real controller would also expose.

Actions are clamped to `[-1, 1]` in `_pre_physics_step` before any of this.

#### Which joints stay independent vs. forced equal

Only 2 of the 16 joints stay fully independent. The other 14 fall into 5 clusters
where every joint is forced to the identical commanded angle (a straight 1:1 copy,
with no ratio or scaling). Worked example using the **mug** variant's tuned values:

| Group | Joints | Tuned per-joint values | After collapsing |
|---|---|---|---|
| (none) | thumb cmc abduction | `1.4` | **Independent**, own actuator |
| (none) | thumb cmc flexion | `0.0` | **Independent**, own actuator |
| thumb tip | mcp / ip | `0.4` / `0.8` | both `0.600` |
| index | mcp / pip / dip | `0.9` / `1.0` / `1.0` | all `0.967` |
| middle | mcp / pip / dip | `0.8` / `0.9` / `0.9` | all `0.867` |
| ring | mcp / pip / dip | `0.9` / `1.0` / `1.0` | all `0.967` |
| pinky | mcp / pip / dip | `0.9` / `1.0` / `1.0` | all `0.967` |

The groupings never change between objects; only the numbers being averaged do. The
collapsed values for all four variants are in
[Per-object reference](#per-object-reference).

A pinch grasp (card, pen) generally wants a larger gap between mcp and pip/dip than a
power grasp (mug, bottle), because a pinch depends on independent fingertip
placement, which is exactly the precision this coupling gives up. That is why the
thin objects are the hard ones to tune by hand.

**Caveat.** This matches TetherIA's *documented* compact representation, which really
is this simple 1:1 duplication for normal use. Their docs also note the real hardware
coupling is more mechanically complex, especially for the thumb: moving one thumb
actuator physically tugs on the others, nonlinearly, rather than producing a clean
group of identical joints. That deeper coupling lives in TetherIA's
`joints_to_actuations.py`, which the *sim* side does not replicate (the
`real_hand/` bridge does; see
[Sim and bridge use different coupling](#sim-and-bridge-use-different-coupling)).
Isaac Lab does not simulate real tendons either, so this is an approximation on top
of an approximation: good enough to train a policy that plays by the same 7-number
rules as the real hand, not a physically exact recreation of the cable mechanics.

The **reset pose** (`grasp_joint_pos`) still specifies all 16 joints individually.
It is written straight into the sim with `write_joint_state_to_sim()`, so it is not
limited by the 7-actuator coupling. Only *ongoing control* during the episode goes
through the 7-dim action space.

### Observations (68-dim)

Layout is identical across every variant; only the values sourced from the object
differ.

| Slice | Size | Source |
|---|---|---|
| Joint positions | 16 | `hand.data.joint_pos` |
| Joint velocities | 16 | `hand.data.joint_vel` |
| Joint torques | 16 | `hand.data.applied_torque` |
| Previous action | 7 | last commanded tendon-space action |
| Object position, relative to hand root | 3 | `object.data.root_pos_w - hand.data.root_pos_w` |
| Object linear velocity | 3 | `object.data.root_lin_vel_w` |
| Object angular velocity | 3 | `object.data.root_ang_vel_w` |
| Object orientation (quaternion) | 4 | `object.data.root_quat_w` |

Returned as `{"policy": obs}`, shape `(num_envs, 68)`.

Two things worth knowing about these:

- `applied_torque` for an **implicit** actuator is Isaac Lab's post-clipping estimate
  computed from the PD error, not a true per-joint torque (PhysX does not expose one
  for implicit drives). It is the closest sim analogue of the real hand's
  per-actuator current sensing, and fine for reward/observation use, but do not treat
  it as a calibrated sensor.
- The object position is relative to the hand's **articulation root**, not a palm
  body or fingertip site. For palm- or fingertip-relative distances (closer to what
  TetherIA's `cube_pos_error` does), print `self.hand.body_names` once in Isaac Sim
  and index into `self.hand.data.body_pos_w`.

### Rewards

Named and structured after TetherIA's reward terms in `rotate_z.py`
(`angvel`/`linvel`/`pose`/`torques`/`energy`/`action_rate`/`termination`), adapted
from "rotate as fast as possible" to "hold still without dropping". Two field names
embed the object's name (`rew_scale_mug_height` in the mug variant,
`rew_scale_bottle_height` in the bottle variant, and so on). **All four variants
currently use the same scale values.**

| Term | Field | Value | Formula |
|---|---|---|---|
| Held / not dropped | `rew_scale_<object>_height` | `5.0` | `scale * held` (boolean → float, every step it hasn't dropped) |
| Stillness | `rew_scale_<object>_still` | `-0.1` | `scale * ‖object_lin_vel‖` |
| Action rate | `rew_scale_action_rate` | `-0.01` | `scale * Σ (action - last_action)²` |
| Pose | `rew_scale_pose` | `-0.05` | `scale * Σ (joint_pos - grasp_joint_pos)²`, over all 16 joints |
| Torques | `rew_scale_torques` | `-1e-3` | `scale * Σ torque²`. **Placeholder, tune.** |
| Energy | `rew_scale_energy` | `-1e-3` | `scale * Σ ‖vel‖ · ‖torque‖`. **Placeholder, tune.** |
| Drop penalty | `rew_scale_drop_penalty` | `-10.0` | `scale * dropped`, every step in the dropped state |

`_get_rewards` reads joint state straight from `hand.data` rather than the cached
`self.joint_pos` / `self.joint_vel`, so it does not depend on whether
`_get_observations` has already run this step. It is also where `self._last_action`
is updated.

### Termination and reset

`_get_dones` returns `(terminated, truncated)`:

- **terminated**: `(initial_height - current_height) > <object>_drop_height_thresh`,
  where height is world `z` minus the env origin's `z`. The initial height is
  captured per env at reset.
- **truncated**: `episode_length_buf >= max_episode_length - 1`, i.e. the 10 s
  episode elapsed.

`_reset_idx` then:

1. Writes the full 16-joint `grasp_joint_pos` and zero velocities into the sim, so
   the hand spawns already closed.
2. Rewrites the hand's root pose and velocity, offset by `scene.env_origins`.
3. Rewrites the object's root pose and velocity, offset by `scene.env_origins`.
4. Records `_initial_<object>_height` for the drop check, and zeroes `_last_action`.

### Scene construction

`_setup_scene` creates the `Articulation` and `RigidObject`, spawns a ground plane at
`/World/ground`, clones the environments, registers them as `scene.articulations["hand"]`
and `scene.rigid_objects["<object>"]`, and adds a dome light at `/World/Light`
(intensity 2000). On CPU it also calls `scene.filter_collisions(global_prim_paths=[])`.

Prim paths are `/World/envs/env_.*/Robot` for the hand and `/World/envs/env_.*/<Object>`
for the object.

---

## Per-object reference

### Object configs

| | mug | bottle | card | pen |
|---|---|---|---|---|
| USD | `only_cup.usda` | `only_bottle.usda` | `only_credit_card.usda` | `only_pen.usda` |
| Mass (kg) | `0.371` | `0.034` | `0.005` | `0.034` |
| Spawn `pos` | `(-0.11, 0.0, 0.85)` | `(-0.12, -0.12, 0.94)` | `(-0.115, -0.02, 0.92)` | `(-0.155, -0.024, 0.965)` |
| Spawn `rot` (w,x,y,z) | identity | `(0, 0, .7071, .7071)` | `(0, .7071, 0, .7071)` | `(0, 0, 1, 0)` |
| Drop threshold (m) | `0.15` | `0.15` | `0.15` | `0.15` |
| Gravity / collision | on / on | on / on | on / on | **off / off** (testing block active) |

The hand root sits at `z = 1.0`, so those spawn heights place the object 15 / 6 / 8 /
3.5 cm below the hand root respectively.

### Tuned grasp poses (radians)

`grasp_joint_pos` is the 16-joint reset pose. The card and pen use only the thumb and
index finger; middle, ring, and pinky stay at `0.0`.

| Joint | mug | bottle | card | pen |
|---|---|---|---|---|
| `thumb_cmc_abd` | 1.4 | 1.5 | 1.25 | 0.5 |
| `thumb_cmc_flex` | 0.0 | 0.1 | 0.1 | 1.9 |
| `thumb_mcp` | 0.4 | 0.7 | 0.7 | 0.4 |
| `thumb_ip` | 0.8 | 1.0 | 1.0 | 0.2 |
| `index_mcp_flex` | 0.9 | 1.1 | 1.6 | 0.4 |
| `index_pip` | 1.0 | 0.6 | 0.55 | 1.6 |
| `index_dip` | 1.0 | 0.6 | 0.6 | 0.4 |
| `middle_*` | 0.8 / 0.9 / 0.9 | 1.1 / 0.6 / 0.6 | 0 | 0 |
| `ring_*` | 0.9 / 1.0 / 1.0 | 1.1 / 0.6 / 0.6 | 0 | 0 |
| `pinky_*` | 0.9 / 1.0 / 1.0 | 1.1 / 0.5 / 0.6 | 0 | 0 |

### Collapsed 7-actuator hold pose

What `test_sim.py` actually commands, after averaging each group (`_grasp_actuation_pos`):

| Actuator | Limit (rad) | mug | bottle | card | pen |
|---|---|---|---|---|---|
| thumb abduction | 1.745 | 1.400 | 1.500 | 1.250 | 0.500 |
| thumb flexion | 0.956 | 0.000 | 0.100 | 0.100 | **1.900 → clamps to 0.956** |
| thumb tip | 1.571 | 0.600 | 0.850 | 0.850 | 0.300 |
| index | 1.571 | 0.967 | 0.767 | 0.917 | 0.800 |
| middle | 1.571 | 0.867 | 0.767 | 0.000 | 0.000 |
| ring | 1.571 | 0.967 | 0.767 | 0.000 | 0.000 |
| pinky | 1.571 | 0.967 | 0.733 | 0.000 | 0.000 |

Three tuned joint values sit **outside** the SDK joint limits and so cannot be
reproduced by the 7-dim controller, only by the direct reset write:

- **pen** `thumb_cmc_flex = 1.9` vs. a limit of `0.956`, the largest violation.
  `test_sim.py` normalizes this to `+2.98`, which `_pre_physics_step` clamps to
  `+1.0`, so the pen's thumb springs from its reset pose to roughly half the
  commanded flexion the moment the first action is applied.
- **card** `index_mcp_flex = 1.6` and **pen** `index_pip = 1.6` vs. a limit of
  `1.5708`. Marginal, but they still make the reset pose impossible to hold exactly.

This is the main reason the card and pen grips currently look unstable in the
viewport: the pose being *held* is not the pose that was tuned. Either bring these
values inside the limits, or accept that a trained policy has to find its own
in-limit grip.

### What is intentionally per-object

1. `<object>_cfg.py`: USD path, mass, spawn pose.
2. `grasp_joint_pos`. A card, a bottle, and a pen need very different finger curl
   from a mug. Tuned by eye with `test_sim.py`.
3. Spawn offset, so the object starts inside the closed grasp for its geometry.
4. `<object>_drop_height_thresh`, which should be sized to the object. Currently
   `0.15` for all four, which is generous for a pen.
5. Reward-scale tuning, especially `rew_scale_torques` / `rew_scale_energy` /
   `rew_scale_pose`. A pinch grasp wants different regularization strength than a
   power grasp. Currently identical across all four.

Everything else (`aero_hand_cfg.py`, the 7-tendon mapping, the observation layout,
and the reward structure) is object-agnostic and identical in all four directories.

### Possible refactor

If maintaining four near-duplicate env files gets annoying, the natural consolidation
is to rename the object-specific identifiers in `aero_hand_grasp_env.py`
(`self.mug` → `self.obj`, `MUG_CFG` → `OBJECT_CFG`, `rew_scale_mug_height` →
`rew_scale_object_height`) and move the object cfg, drop threshold, and reward scales
into fields on a single shared `AeroHandGraspEnvCfg`, giving one env class and four
cfg instances.

---

## Running on the real hand

Everything that talks to the physical hand over ROS2 lives in `real_hand/`, and
nothing else does. It is object-agnostic (the hand does not care what it is holding),
so it sits at the top level rather than being copied into each variant.

| File | What it is |
|---|---|
| `real_hand/aero_hand_bridge.py` | Abstraction between 7 actuator values (radians) and the real hand over ROS2. Hides joint-space expansion, unit conversion, and ROS2 plumbing. Also reconstructs joint positions from motor angles by inverting the tendon model. |
| `real_hand/test_ros.py` | Choreographed demo: finger wave, counting 1→5, rock on, thumbs up, thumb opposition, piano taps, slow clench. Poses are interpolated and streamed at 100 Hz. Doubles as an end-to-end check that commands and feedback round-trip. |

Variant scripts run from their own directory, so import the bridge with:

```python
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "real_hand"))
from aero_hand_bridge import AeroHandBridge
```

### One-time setup

Requires ROS2 Humble, the `aero-hand-open` workspace built with `colcon`, and the
`aero_open_sdk` Python package installed.

**The SDK needs a patch to talk to the hand over USB.** The Aero Hand enumerates as
an ESP32-S3 native USB-JTAG-Serial device, where **RTS drives the chip's reset
line**, and pyserial asserts both DTR and RTS when it opens a port. The chip
therefore sits held in reset and never answers, and every command dies with:

```
ACK (opcode 0x31) not received within 2.0s
```

Opening with RTS already low does *not* fix it; the chip has to actually see the
reset released, so the transition is what matters. `AeroHand.__init__` needs a wake
sequence before its first command:

```python
self.ser.dtr, self.ser.rts = True, True
time.sleep(0.3)
self.ser.dtr, self.ser.rts = False, False   # release EN -> chip boots
time.sleep(0.3)
self.ser.dtr, self.ser.rts = True, False    # run mode
time.sleep(1.0)                             # let it boot before talking
```

Baudrate is irrelevant here, since it is native USB-CDC: 115200 behaves identically
to 921600.

The SDK installs as a **non-editable copy** into site-packages by default, so editing
the repo has no effect until you re-point the install:

```bash
pip install -e /path/to/aero-hand-open/sdk
```

### Starting the hardware node

```bash
ros2 run aero_hand_open aero_hand_node --ros-args \
    -p right_port:=auto -p control_space:=joint
```

`right_port:=auto` resolves through `/dev/serial/by-id/usb-Espressif_*`, which is the
robust choice, because the wake sequence resets the chip: it re-enumerates and the
`/dev/ttyACM<N>` number changes. Never hardcode `/dev/ttyACM0`.

**Only run one node at a time.** A second instance resets the chip out from under the
first, which then holds a dead device node.

The bridge assumes this node is up and publishing on `/{side}/actuator_states` and
subscribing on `/{side}/joint_control` (`side` defaults to `"right"`).

### Network isolation

If you have a FastDDS unicast profile (`FASTRTPS_DEFAULT_PROFILES_FILE`) pointing
discovery at another machine, you will see *that* machine's hand instead of your own,
including being able to command it. The symptom is a topic list that looks correct
while no data ever arrives, because `avoid_builtin_multicast` plus an
`initialPeersList` aimed elsewhere lets participants match over shared memory without
endpoint discovery completing:

```
sequence size exceeds remaining buffer
RuntimeError: No feedback from hand — is aero_hand_node running?
```

For local-only operation:

```bash
unset FASTRTPS_DEFAULT_PROFILES_FILE
export ROS_LOCALHOST_ONLY=1
ros2 daemon stop        # the daemon caches discovery config
```

`.bashrc` makes this the default, with the remote peer available via `AERO_REMOTE=1`.
Shell startup files only apply to **newly opened** terminals, so a stale terminal is
the usual cause of this failing after it was "fixed."

### The bridge API

```python
with AeroHandBridge() as hand:                       # AeroHandBridge(side="right")
    hand.send_actuator_positions([0.0, 0.3, 0.35, 0.6, 0.6, 0.6, 0.6])
    hand.spin_once(timeout_sec=0.0)

    actuators = hand.get_actuator_feedback()           # 7,  radians, same space as the command
    joints    = hand.get_joint_feedback()              # 16, radians, JOINT_NAMES order
    motors    = hand.get_motor_feedback()              # 7,  radians, raw hardware
    speeds    = hand.get_actuator_speed_feedback()     # 7,  RPM
    currents  = hand.get_actuator_current_feedback()   # 7,  mA, torque proxy
```

Every getter returns `None` until the first `ActuatorStates` message arrives; use
`hand.wait_for_feedback(timeout_sec=5.0)` (returns `bool`) to block for it. The
bridge initializes `rclpy` only if it is not already running, and shuts it down again
only if it initialized it, so it composes with an existing ROS2 node. All reads are
mutex-guarded, so you can spin on one thread and read from another.

`ACTUATOR_NAMES` on the bridge uses hardware names
(`thumb_abduction_actuator`, `thumb_flex_actuator`, `thumb_tendon_actuator`,
`index_finger_actuator`, `middle_finger_actuator`, `ring_finger_actuator`,
`pinky_finger_actuator`) rather than the sim's `right_*_act` names, but **the order is
the same**, which is what makes a sim action directly sendable.

Per-actuator travel, derived from the USD joint limits divided by the coupling ratios
(`send_actuator_positions` clamps to these, silently):

| Actuator | Index | Upper limit (rad) |
|---|---|---|
| `thumb_abduction_actuator` | 0 | 1.745 |
| `thumb_flex_actuator` | 1 | 0.956 |
| `thumb_tendon_actuator` | 2 | 2.618 |
| `index` / `middle` / `ring` / `pinky` | 3 to 6 | 1.571 |

### Sim and bridge use different coupling

This matters and is easy to miss. Both sides map 7 actuators to 16 joints, but with
**different ratios**:

| | sim (`ACTUATOR_TO_JOINTS`) | bridge (`ACTUATOR_JOINT_MAP`) |
|---|---|---|
| thumb tendon → mcp, ip | `1.0`, `1.0` | `0.6`, `0.6` |
| finger → mcp, pip, dip | `1.0`, `1.0`, `1.0` | `1.0`, `0.6`, `0.6` |
| thumb abd / flex | `1.0` | `1.0` |

The bridge's ratios are the ones the hardware's tendon model actually implements; the
sim's 1:1 broadcast is TetherIA's simplified "compact representation". So the same
7 numbers produce a *differently shaped* hand in sim than on hardware: the fingertips
curl less on the real hand. That is a known sim-to-real gap, not a bug in either
file, but it is the first thing to check if a transferred policy grips differently
than it did in sim. Closing it means changing the sim's broadcast matrix from ones to
the bridge's ratios, and re-tuning `grasp_joint_pos` afterwards.

### How joint feedback is reconstructed

**The hardware has no joint encoders.** It publishes only `ActuatorStates`, which is
7 raw *motor* angles in degrees, and the SDK's `get_joint_positions()` is an
unimplemented stub. The bridge therefore mirrors the SDK's `joints_to_actuations`
model locally (so it stays dependency-free) and inverts it:

1. `TENDON_COEFFS`: mm/radian coefficients per actuator, with
   `MOTOR_PULLEY_RADIUS = 9.0 mm`, giving
   `motor[a] = Σ_j coeff[a][j] · joint[j] / pulley_radius`. The thumb rows are
   cross-coupled: `thumb_flex_actuator` depends on `thumb_cmc_abd`, and
   `thumb_tendon_actuator` on both `thumb_cmc_abd` and `thumb_cmc_flex` (with an
   intentionally negative sign).
2. `_build_motor_matrix` composes that with the actuator-to-joint coupling into a 7×7
   motor matrix. Because of the thumb cross-coupling it is **lower-triangular**, not
   diagonal, which is exactly why feedback cannot be a per-actuator scale factor.
   The builder asserts both the triangularity and a non-zero diagonal at construction
   and raises rather than returning quietly wrong angles if someone edits
   `ACTUATOR_JOINT_MAP` into an incompatible shape.
3. `_motor_to_actuator` inverts it by forward substitution, then
   `_actuator_to_joints` expands back out to 16 joints.

Verified against the SDK's own forward model with a worst round-trip error of
**1.1e-16**, but it is still a *model estimate*: tendon stretch, and an object
physically blocking a finger, are both invisible to it. For contact detection use
`get_actuator_current_feedback()`, which is real sensing.

Motor space and actuator space are **not** interchangeable: the finger tendon gain
is `22.28386 / 9.0 ≈ 2.476`, so commanding `0.8` reads back `1.9592` in raw motor
radians. `get_actuator_feedback()` returns the value directly comparable to what you
sent; `get_motor_feedback()` returns the raw reading. Expect tracking residuals of
0.5 to 2% on the fingers and up to ~0.056 rad on the thumb, which is ordinary servo
error and tendon compliance rather than a units problem. The thumb is worse because
it is the cross-coupled chain with the tightest limit.

### Control rates

| | Rate | Where it is set |
|---|---|---|
| Real hand sensing | **100 Hz** | `feedback_frequency` param; firmware sync-read task at `pdMS_TO_TICKS(10)` |
| Real hand commands | event-driven, unthrottled | written on every `JointControl` message |
| Sim policy | **60 Hz** | `sim.dt = 1/120` with `decimation = 2` |

Measured 100.01 Hz feedback, holding steady even while commanding at 800 Hz. That
measures the ROS/USB path, *not* servo actuation: the node publishes whatever is in
the firmware's `gMetrics` buffer, so starved servo readings would go stale without the
publish rate ever dropping. Commands and sensing share a 1 Mbps servo bus behind a
mutex, and the read task uses a **try-lock** that skips its cycle when control holds
the bus, so commands have priority over sensing by design. There is nothing to gain
above 100 Hz.

The 60 Hz sim vs. 100 Hz hardware gap matters for sim-to-real. Align them with
`decimation = 1` (120 Hz) or by dropping `feedback_frequency` to 60; commanding a
100 Hz hand at 60 Hz is otherwise fine, since commands latch until the next one.

### Demo

```bash
python3 real_hand/test_ros.py
```

Roughly 25 s: open palm, finger wave, counting 1→5, rock on, thumbs up, thumb
opposition across all four fingers, piano taps, and a slow clench. It requires
feedback within 5 s of start (`wait_for_feedback`) and otherwise raises
`RuntimeError: No feedback from hand — is aero_hand_node running?`.

Poses are 7 actuator values in radians. Motion between them is interpolated with
`smoothstep` easing (zero velocity at both ends) and streamed at 100 Hz to match the
hand's own loop, rather than sent as step targets the servos have to chase. Stepping
straight to a target looks jerky and slams the tendons. `Ctrl+C` glides back to an
open palm instead of freezing mid-pose. At the end it prints commanded vs. measured
actuator positions with the residual, plus all 16 reconstructed joint angles.

The `PINCH` triples (thumb abduction/flexion/curl per finger) are geometric estimates
of fingertip contact, not measured. Trim them if a pinch misses or collides on your
unit.

---

## Known placeholders: check before training

**Sim, shared across all four variants:**

- `aero_hand_cfg.py`: `stiffness = 15.0` and `effort_limit_sim = 2.0` are
  placeholders that have not been matched to the real servo response.
- `aero_hand_cfg.py`: the base rotation quaternion is `160°` about `+Y` while the
  inline comment claims `20°` about `+X`. Verify by eye which one you actually want.
- `aero_hand_grasp_env.py` `_apply_action()`: **remove the debug print block before
  training.** It prints the full 7-actuator control signal for env 0 on every physics
  substep (twice per `env.step()` at `decimation = 2`), and every `.item()` /
  `.tolist()` in it forces a GPU→CPU sync.
- `rew_scale_torques` and `rew_scale_energy` are `-1e-3` guesses in all four
  variants; verify against actual torque magnitudes once you can log them, per object.
- `<object>_drop_height_thresh` is `0.15` in all four variants; size it per object.

**Per object:**

- `pen_cfg.py` currently has the **testing block active**: gravity and collision are
  both disabled. Swap back to the real block before training or evaluating.
- `bottle_cfg.py` mass is `0.034 kg`, the same value as the pen, almost certainly a
  copy-paste that was never updated.
- Card and pen `grasp_joint_pos` contain values outside the SDK joint limits (see
  [Collapsed 7-actuator hold pose](#collapsed-7-actuator-hold-pose)); the pen's
  `thumb_cmc_flex = 1.9` against a `0.956` limit is the significant one.
- Object masses and spawn `pos` for all four are eyeballed, not measured.

**Cross-cutting:**

- The sim and bridge coupling ratios differ (see
  [Sim and bridge use different coupling](#sim-and-bridge-use-different-coupling)).
  Decide which side to change before transferring a policy.
- `aero_hand_scene_cfg.py` is dead code in all four directories; nothing imports it.

---

## Troubleshooting

### Simulation

| Symptom | Cause |
|---|---|
| USD asset fails to resolve / `aero_hand.usda` not found | Script run from the repo root instead of from inside the object directory; the USD paths are `../hand_model/...` relative to the working directory |
| Object falls straight through the hand | The pen variant ships with collision disabled; check whether the "for testing" block in `<object>_cfg.py` is the active one |
| Object never falls, ever | Same block: gravity disabled |
| Grip visibly snaps to a different pose on the first step | The tuned `grasp_joint_pos` collapses to an actuator target outside its limit and is being clamped; see the card/pen note above |
| Console flooded with `_apply_action() debug` output | The debug block in `_apply_action` is still enabled; delete it |

### Hardware

| Symptom | Cause |
|---|---|
| `ACK (opcode 0x31) not received` | SDK missing the DTR/RTS wake sequence, or SDK installed non-editable so the patch isn't live |
| `No feedback from hand` + `sequence size exceeds remaining buffer` | Stale terminal still has the FastDDS unicast profile; open a new shell and `ros2 daemon stop` |
| Topics listed but no data | Same as above: participants match over shared memory while endpoint discovery never completes |
| Node starts, then reads fail | Two nodes running; the second reset the chip and it re-enumerated to a new `/dev/ttyACM<N>` |
| Feedback ~2.5× the commanded value | Reading `get_motor_feedback()` where `get_actuator_feedback()` was meant |
| Hand not detected at all after plugging in | USB-C cable is charge-only; use one that carries data |
| `Motor matrix is not lower-triangular` at bridge startup | `ACTUATOR_JOINT_MAP` was edited into a shape the forward-substitution inverse can't handle |

---

## Website

`website/` is a static documentation site: `index.html` and `index-v2.html`, plus
`public/` (compiled `styles.css` from `styles.scss`, `index.js` for the sticky navbar
and hamburger menu, and `img/` render screenshots of each object). `index-v2.html` is
the current version and tracks this README. Open either file directly in a browser;
no build step is required to view them, though editing styles means recompiling the
SCSS.

## License

See [LICENSE](LICENSE).
