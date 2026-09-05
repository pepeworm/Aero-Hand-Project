# bottle_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

BOTTLE_USD_PATH = "../hand_model/only_bottle.usda"

BOTTLE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BOTTLE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.034),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.12, -0.12, 0.94),   # PLACEHOLDER — tune alongside grasp_joint_pos
        rot=(0.0, 0.0, 0.70710678, 0.70710678),
    ),
)

# For testing purposes (turns off gravity and collision)
# import isaaclab.sim as sim_utils
# from isaaclab.assets import RigidObjectCfg

# BOTTLE_USD_PATH = "../hand_model/only_bottle.usda"

# BOTTLE_CFG = RigidObjectCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=BOTTLE_USD_PATH,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,
#             max_depenetration_velocity=1.0,
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(0.034),

#         # disable collision for testing purposes
#         collision_props=sim_utils.CollisionPropertiesCfg(
#             collision_enabled=False
#         ),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=(-0.12, -0.12, 0.94),   # PLACEHOLDER — tune alongside grasp_joint_pos
#         rot=(0.0, 0.0, 0.70710678, 0.70710678),
#     ),
#)