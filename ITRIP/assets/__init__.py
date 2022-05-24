from pathlib import Path
import numpy as np
import os

cur_path = Path(__file__).parent

UR5_SCENE_FILE_PATH = str(cur_path.joinpath('scene_ur5_suction.ttt'))
UR5_SCENE_RANDOM_FILE_PATH = str(cur_path.joinpath('scene_ur5_suction_randomTexture.ttt'))
UR5_GRASPNET_SCENE_RANDOM_FILE_PATH = str(cur_path.joinpath('scene_GraspNet_ur5_suction_randomTexture.ttt'))
UR5_SCENE_GRASPNET_FILE_PATH = str(cur_path.joinpath('scene_ur5_suction_GraspNet.ttt'))

EMPTY_BASKET_DEPTH = np.load(str(cur_path.joinpath('empty_basket_depth.npy')))

GRASPNET_TTM_ROOT = str(cur_path.joinpath('GraspNet_models'))

UR5_SCENE_GRASPNET_PICKING_FILE_PATH = str(cur_path.joinpath('scene_ur5_suction_GraspNet_Picking.ttt'))
UR5_SCENE_ITRI_PICKING_FILE_PATH = str(cur_path.joinpath('scene_ur5_suction_ITRI_Picking.ttt'))
UR3_LAB_OBJ_SCENE_FILE_PATH = str(cur_path.joinpath('scene_ur3_suction_lab_obj.ttt'))
UR5_LAB_OBJ_SCENE_FILE_PATH = str(cur_path.joinpath('scene_ur5_suction_lab_obj.ttt'))

VACUUM_PICK_SCRIPT = str(cur_path.joinpath('vacuum_pick.script'))
VACUUM_CHECK_SCRIPT = str(cur_path.joinpath('vacuum_check.script'))
VACUUM_RELEASE_SCRIPT = str(cur_path.joinpath('vacuum_release.script'))


#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
# Deprecated
# get models
# MODEL_PATHS = []
# model_root_path = cur_path.joinpath('models')
# for fdr in os.listdir(model_root_path):
#     path = model_root_path.joinpath(fdr)
#     objs_path = list(path.glob("*.obj"))
#     # col_path = [p for p in objs_path if 'col' in p.name][0]
#     vis_path = [p for p in objs_path if 'col' not in p.name][0]

#     MODEL_PATHS.append({
#         # 'col': col_path,
#         'vis': vis_path
#     })

# del model_root_path