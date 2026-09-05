# card_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

CARD_USD_PATH = "../hand_model/only_credit_card.usda"

CARD_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CARD_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.115, -0.02, 0.92),  # PLACEHOLDER — tune alongside grasp_joint_pos
        rot=(0.0, 0.70710678, 0.0, 0.70710678),
    ),
)

# # For testing purposes (turns off gravity and collision)
# import isaaclab.sim as sim_utils
# from isaaclab.assets import RigidObjectCfg

# CARD_USD_PATH = "../hand_model/only_credit_card.usda"

# CARD_CFG = RigidObjectCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=CARD_USD_PATH,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             max_depenetration_velocity=1.0,
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(0.005),

#         # disable collision for testing purposes
#         collision_props=sim_utils.CollisionPropertiesCfg(
#             collision_enabled=False
#         ),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=(-0.11, -0.02, 0.92),   # PLACEHOLDER — tune alongside grasp_joint_pos
#         rot=(0.0, 0.70710678, 0.0, 0.70710678),
#     ),
# )