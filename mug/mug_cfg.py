# mug_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

MUG_USD_PATH = "../hand_model/only_cup.usda"

MUG_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MUG_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.371),   # PLACEHOLDER — set to your real mug's mass
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.11, 0.0, 0.85),
    ),
)   

# # For testing purposes (turns off gravity and collision)
# import isaaclab.sim as sim_utils
# from isaaclab.assets import RigidObjectCfg

# MUG_USD_PATH = "../hand_model/only_cup.usda"

# MUG_CFG = RigidObjectCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=MUG_USD_PATH,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             max_depenetration_velocity=1.0,
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(mass=0.371),

#         # disable collision for testing purposes
#         collision_props=sim_utils.CollisionPropertiesCfg(
#             collision_enabled=False
#         ),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=(-0.11, 0.0, 0.85),
#     ),
# )
