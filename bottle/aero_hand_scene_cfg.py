# aero_hand_scene_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from aero_hand_cfg import AERO_HAND_CFG


@configclass
class AeroHandSceneCfg(InteractiveSceneCfg):
    """N side-by-side Aero Hands, cloned via Isaac Lab's GPU-parallel cloner."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # {ENV_REGEX_NS} is replaced with /World/envs/env_{i} per environment
    aero_hand: ArticulationCfg = AERO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
