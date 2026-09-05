# aero_hand_cfg.py
"""Configuration for the TetherIA Aero Hand Open (right hand)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

AERO_HAND_USD_PATH = (
    "../hand_model/aero_hand.usda"
)

AERO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=AERO_HAND_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,   # fingers can touch palm/each other
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),   # lift off the ground plane
        rot=(0.173648, 0.0, 0.984807, 0.0),  
        joint_pos={
            "right_index_mcp_flex": 0.0,
            "right_index_pip": 0.0,
            "right_index_dip": 0.0,
            "right_middle_mcp_flex": 0.0,
            "right_middle_pip": 0.0,
            "right_middle_dip": 0.0,
            "right_ring_mcp_flex": 0.0,
            "right_ring_pip": 0.0,
            "right_ring_dip": 0.0,
            "right_pinky_mcp_flex": 0.0,
            "right_pinky_pip": 0.0,
            "right_pinky_dip": 0.0,
            "right_thumb_cmc_abd": 0.0,
            "right_thumb_cmc_flex": 0.0,
            "right_thumb_mcp": 0.0,
            "right_thumb_ip": 0.0, 
        },
    ),
    actuators={
        "finger_joints": ImplicitActuatorCfg(
            joint_names_expr=["right_.*"],   # matches all 16 joints
            stiffness=15.0,    # PLACEHOLDER — tune against real servo response, #TODO change this
            damping=1.0,       # same as you tuned the actuator ratios earlier
            effort_limit_sim=2.0,
            velocity_limit_sim=5.0,
        ),
    },
)
