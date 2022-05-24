from operator import xor
from typing import Dict, Tuple
from numpy.core.fromnumeric import clip
from numpy.core.function_base import linspace
from pyrep import PyRep
from pyrep.robots.arms.ur5 import UR5
# from pyrep.robots.end_effectors.baxter_suction_cup import BaxterSuctionCup
from pyrep.objects.vision_sensor import VisionSensor
from tqdm import tqdm
import numpy as np
import itertools
import quaternion

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import open3d as o3d

from ITRIP.baseEnv import SuctionBaseEnv
from ITRIP.assets import UR5_SCENE_FILE_PATH,UR5_SCENE_RANDOM_FILE_PATH
from ITRIP.utils import center_scaling,decode_bin, encode_bin,\
    m_euler2quat as euler2quat, m_quat2euler as quat2euler, wxyz2xyzw, xyzw2wxyz
from ITRIP.objects import ObjectLoader
from ITRIP.SteroidSuctionCup import SteroidBaxterSuctionCup as BaxterSuctionCup

import time
import cv2
from os import path
import unittest
from pyrep.backend.simConst import *
from pyrep.objects import Shape
from pyrep.const import  *
from pyrep.textures.texture import Texture
from pyrep.objects.dummy import Dummy
from DON_Picking.Configuration import *


UR5_PICK_JPOS = [0, 0, -np.pi/2, 0, np.pi/2, 0]
UR5_IDLE_JPOS = [np.pi/2, 0, -np.pi/2, 0, np.pi/2, 0]
NORM_SAMPLE_RANGE = 20
MAX_TILT_ANG = 60
IK_ITERS = 100
MIN_PICK_DIST2BASKET = 0.002
MAX_STOP_WAIT = 30
MAX_EXECUTE_TIME = 2 #second

class SimSuctionEnv(SuctionBaseEnv):

    def __init__(self, envIdx = 0, img_res=(256,256), headless=False, debug=False, exclude=[], isEvaluation = False) -> None:
        super().__init__(img_res, None)

        np.random.seed((int)(time.time())+ envIdx)

        self.debug = debug
        
        self.pr = PyRep()
        self.pr.launch(UR5_SCENE_RANDOM_FILE_PATH, headless=headless)
        self.pr.start()

        # set up control for robot
        self.arm = UR5()
        self.suction_cup = BaxterSuctionCup()
        self.suction_cup_body = Shape('BaxterSuctionCup_body')

        # read ini locations
        # self._ini_jpos = np.array(self.arm.get_joint_positions())
        self._ini_pos, self._ini_euler = np.array(self.arm.get_tip().get_position()), \
                        np.array(self.arm.get_tip().get_orientation())
        self._init_cup_pos, self._init_cup_euler = np.array(self.suction_cup.get_position()), \
                        np.array(self.suction_cup.get_orientation())
        self._init_cup_body_pos, self._init_cup_body_euler = np.array(self.suction_cup_body.get_position()), \
                        np.array(self.suction_cup_body.get_orientation())

        self._ini_arm_config = self.arm.get_configuration_tree()
        self._ini_suction_cup_config = self.suction_cup.get_configuration_tree()

        # set up sensors
        # self.hand_proxi_sensor = ProximitySensor('Hand_proximity')
        self.hand_cam = VisionSensor('Hand_cam')
        self.basket_cam = VisionSensor('Basket_cam')

        self.hand_cam.set_resolution(img_res)
        self.basket_cam.set_resolution(img_res)

        self.objects = []
        self.cam_intrinsics = {
            'hand': self.hand_cam.get_intrinsic_matrix(),
            'basket': self.basket_cam.get_intrinsic_matrix(),
        }
        self.cams = {
            'hand': self.hand_cam,
            'basket': self.basket_cam,
        }

        # set up object loading
        
        try:
            self.dummy_origin = Dummy("Dummy_Origin")
            self.plane_background = Shape("Plane_Background")
        except:
            pass

        self.obj_loader = ObjectLoader(exclude=exclude)
        crate = Shape('crate_visible')
        self.crate = crate

        #self.crate.set_renderable(True)
        self.crate.set_renderable(False)
        min_x, max_x, min_y, max_y, min_z, max_z = crate.get_bounding_box()
        self.sample_max_dist = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2

        min_z = max_z
        max_z += 0.3

        min_x, max_x = center_scaling(min_x, max_x, 0.7)
        min_y, max_y = center_scaling(min_y, max_y, 0.7)
        # min_z, max_z = center_scaling(min_z, max_z, 0.9)

        self.crate_pos = np.array(crate.get_position())
        self.obj_rand_lowers = np.array([min_x, min_y, min_z]) + self.crate_pos
        self.obj_rand_highers = np.array([max_x, max_y, max_z]) + self.crate_pos
        # self.obj_rand_lowers = np.array([min_x, min_y, min_z])
        # self.obj_rand_highers = np.array([max_x, max_y, max_z])
        self._crate_bbox = np.array(crate.get_bounding_box()).reshape((3, 2)).T + self.crate_pos
        self._crate_bbox = self._crate_bbox.T.flatten()
        self._crate_bbox[-1] += 0.3

        self._empty_basket_pcd = None

        self.errorCode = DONE_NONE
        self.isCrashed = False
        self.totalExecuteTime = 0
        self.totalReward = 0
        self.countPickedObject = 0
        self.failedAttemp = 0

        self.isEvaluation = isEvaluation

        # debug stuff
        if self.debug:
            self.dummy_sphere = Shape.create(type=PrimitiveShape.SPHERE,
                                         color=[1, 0, 0],
                                         size=[0.05] * 3,
                                         respondable=False,
                                         static=True)

    def stop(self):
        self.pr.stop()
        self.pr.shutdown()

    def wait_for_stop(self):
        for _ in range(MAX_STOP_WAIT):
            move = False
            for obj in self.objects:
                linear_vel, angular_vel = obj.get_velocity()
                linear_speed = np.linalg.norm(linear_vel)
                angluar_speed = np.max(angular_vel)
                if linear_speed > 0.01 or angluar_speed > 1:
                    move = True
                    break
            if move:
                self.pr.step()
            else:
                break

    def reset(self, num_objs=10, hand_obs=False, basket_obs=True,objectID = None,randomTexture= False) -> Tuple[int, Dict]:

        # reset variable
        self.errorCode = DONE_NONE
        self.isCrashed = False
        self.totalExecuteTime = 0
        self.totalReward = 0
        self.countPickedObject = 0
        self.failedAttemp = 0
        self.objectOut=0
        self.lastNumberAction = 0

        self.MAX_ATTEMP_ACTION = num_objs * 2

        self._reset_arm()
        self.steps = 0
        
        for obj in self.objects:
            obj.remove()

        self.pr.step()
        self.basket_cam.handle_explicitly()
        self._empty_basket_pcd = self.basket_cam.capture_pointcloud()

        self.objects = []
        # random place objects
        for _ in np.arange(num_objs):

            if (objectID is not None):
                obj = self.obj_loader.get_random_obj(idx=objectID)
            else:
                obj = self.obj_loader.get_random_obj(randomTexture=randomTexture)

            rot, pos = self._sample_random_obj_pose()
            obj.set_orientation(rot)
            obj.set_position(pos)

            self.pr.step()

            self.objects.append(obj)

        self.wait()

        self.wait_for_stop()

        # re-drop outside objects
        for obj in self.objects:
            for _ in range(5):
                obj_pos = np.array(obj.get_position())
                # obj_dist = np.linalg.norm(obj_pos - self.crate_pos)

                # if obj_dist <= self.sample_max_dist:
                if self._check_within_crate(obj_pos):
                    break

                rot, pos = self._sample_random_obj_pose()
                obj.set_orientation(rot)
                obj.set_position(pos)

                self.pr.step()

        self.wait()
        self.wait_for_stop()

        self._remove_outside_objects()

        # obs = self._take_selected_obs(hand_obs, basket_obs)

        #if (self.isEvaluation):
        if (len(self.objects) !=  num_objs):
            self.reset()

        #only once for training
        #self.changeRandomTexture()

        return len(self.objects)

    def changeRandomTexture(self):
        for obj in self.objects:
            texture,mode = self.obj_loader.getRandomTexture()
            obj.changeTexture(texture,mode)

    def changeRandomBackground(self):
        r,g,b = random.randint(0,255)/255.0, random.randint(0,255)/255.0, random.randint(0,255)/255.0
        self.plane_background.set_color([r,g,b])

    def freezeObject(self):
        for obj in self.objects:
            obj.set_dynamic(False)
            obj.set_respondable(False)
            self.pr.step()

    def wait (self):
        for _ in range (50):
            self.pr.step()

    def getRemainingObject(self):
        return len(self.objects)

    def isTerminal(self):
        #crash
        if (self.isCrashed):
            self.errorCode = DONE_CRASH
            return True

        #grab all objects
        if (self.done()):
            self.errorCode = DONE_FINISH
            return True

        if (self.steps > self.MAX_ATTEMP_ACTION):
            self.errorCode = DONE_EXCEEDED_MAX_ACTION
            return True

        return False

    def _check_within_crate(self, pos):
        x, y, z = pos
        min_x, max_x, min_y, max_y, min_z, max_z = self._crate_bbox

        return min_x <= x <= max_x and \
               min_y <= y <= max_y and \
               min_z <= z <= max_z

    def _remove_outside_objects(self, whitelist=[]):
        inside_objects = []
        detect_outside = False

        for obj in self.objects:
            if obj in whitelist:
                continue

            obj_pos = np.array(obj.get_position())

            within_crate = self._check_within_crate(obj_pos)

            # obj_dist = np.linalg.norm(obj_pos - self.crate_pos)
            # within_crate = obj_dist <= self.sample_max_dist

            if within_crate:
                inside_objects.append(obj)
            else:
                obj.remove()
                detect_outside = True

        self.objects = inside_objects

        return detect_outside

    def _sample_random_obj_pose(self):
        rot = np.random.uniform(0, 2*np.pi, 3)
        pos = np.random.uniform(self.obj_rand_lowers, self.obj_rand_highers)

        return rot, pos

    def _set_jpos(self, jpos, robust=True):
        self.arm.set_joint_positions(jpos, disable_dynamics=True)
        self.arm.set_joint_target_positions(jpos)

        if robust:
            self._move2jpos(jpos)

    def _take_obs(self, cam: VisionSensor):
        cam.handle_explicitly() # explicit handling for faster simulation

        depth = cam.capture_depth(in_meters=True)
        rgb = cam.capture_rgb()



        # make into 255 int for memory efficiency
        #rgb = rgb * 255
        #rgb = rgb.astype(np.uint8)
        #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        pcd = cam.pointcloud_from_depth(depth)

        pcd_d_diff = pcd[:, :, -1] - self._empty_basket_pcd[:, :, -1]
        # pass in terms of valid actions since there are fewer of them
        valid_actions = pcd_d_diff >= MIN_PICK_DIST2BASKET
        valid_actions_enc = encode_bin(valid_actions)

        self.valid_actions_enc = valid_actions_enc
        depth = depth.reshape(config["W"], config["W"], 1)
        self.depth = depth
        self.rgb = rgb
        valid_actions_decode = decode_bin(valid_actions_enc)

        if (np.max(valid_actions_decode) ==0):
            valid_actions_decode[config["HalfWidth"],config["HalfWidth"]] = 1

        valid_actions_decode =  valid_actions_decode.reshape(config["W"], config["W"], 1)
        
        return depth, rgb ,valid_actions_decode
    '''
    def _take_selected_obs(self, hand_obs, basket_obs):
        self.wait_for_stop()
        obs = []
        if hand_obs:
            obs.append(self._take_obs(self.hand_cam))
        if basket_obs:
            obs.append(self._take_obs(self.basket_cam))
        if len(obs) == 1:
            return obs[0]
        return obs
    '''

    def _take_selected_obs(self, hand_obs, basket_obs):
        self.wait_for_stop()
        if (basket_obs):
            return self._take_obs(self.basket_cam)
        else:
            return self._take_obs(self.hand_cam)

    def checkCrash(self, accumulateTime):
        self.totalExecuteTime += accumulateTime
        if (self.totalExecuteTime > MAX_EXECUTE_TIME):
            self.errorCode = DONE_CRASH
            self.isCrashed = True
            #print ("CRASH",self.totalExecuteTime)
            self.totalExecuteTime = 0
            return True
        return False

    def _move2jpos(self, pos, stop_at_grasp=False):
        tg_pos = np.array(pos)

        self.arm.set_joint_target_positions(tg_pos)

        idle_cnt = 0
        max_cnt = 0
        last_jpos = np.array(self.arm.get_joint_positions())
        self.totalExecuteTime = 0
        startTime = time.time()
        while np.any(np.abs(last_jpos - tg_pos) > 0.0001) and idle_cnt < 10 and max_cnt < 30:
            self.pr.step()
            '''
            executeTime = time.time() - startTime
            if self.checkCrash(executeTime):
                return False
            '''
            # check loop callback stop
            if stop_at_grasp:
                obj = self._grasp()

                if obj is not None:
                    return obj

            cur_jpos = np.array(self.arm.get_joint_positions())

            if not np.any(np.abs(last_jpos - cur_jpos) > 0.001):
                idle_cnt += 1
            else:
                idle_cnt = 0

            last_jpos = cur_jpos
            max_cnt += 1

        if stop_at_grasp:
            return None
        '''
        if (max_cnt >=30 ):#or idle_cnt >=10):
            self.errorCode = DONE_CRASH
            self.isCrashed = True
            return False
        '''

    def _move2pose(self, pos, euler=np.array([0, 0, 0]), teleport=False, stop_at_grasp=False):
        assert not (xor(stop_at_grasp, stop_at_grasp) and (teleport or stop_at_grasp)), "Only teleport or stop_at_grasp"

        tg_jpos = None
        def_euler = np.array([0, 0, 0])

        # tg_jpos = self.arm.solve_ik_via_jacobian(pos, euler=euler)

        # linear interpolation in angle
        # for i in range(IK_ITERS + 1):
        #     try:
        #         t_euler = (def_euler - euler) * i / IK_ITERS + euler
        #         tg_jpos = self.arm.solve_ik_via_jacobian(pos, euler=t_euler)
        #         break
        #     except Exception:
        #         pass

        # slerping to solve IK
        def_quat = np.quaternion(*xyzw2wxyz(euler2quat(*np.array([0, 0, np.pi]))))
        m_quat = np.quaternion(*xyzw2wxyz(euler2quat(*euler)))
        for i in range(IK_ITERS + 1):
            try:
                t_quat = quaternion.slerp(m_quat, def_quat, 0, 1, i / IK_ITERS).components
                t_quat = wxyz2xyzw(t_quat)
                tg_jpos = self.arm.solve_ik_via_jacobian(pos, quaternion=t_quat)
                break
            except Exception as e:
                pass

        if tg_jpos is None:
            try:
                tg_jpos = self.arm.solve_ik_via_sampling(pos, euler=euler)[0]
            except Exception:
                pass

        if tg_jpos is None:
            tg_jpos = self.arm.solve_ik_via_jacobian(pos, euler=def_euler)

        if teleport:
            self._set_jpos(tg_jpos)
        else:
            res = self._move2jpos(tg_jpos, stop_at_grasp=stop_at_grasp)

            if res is None:
                res = self._grasp()
            return res

    def _reset_arm(self):
        self.pr.set_configuration_tree(self._ini_arm_config)
        self.pr.set_configuration_tree(self._ini_suction_cup_config)
        self._set_jpos(UR5_IDLE_JPOS)
        self.suction_cup.release()

    def take_obs(self, hand_obs=False, basket_obs=True) -> Tuple[np.array, np.array]:
        self._reset_arm()
        self.pr.step()

        obs = self._take_selected_obs(hand_obs, basket_obs)

        return obs

    def done(self) -> bool:
        return len(self.objects) == 0

    def _normal_from_pcd(self, pcd, x, y):
        x_dim, y_dim = pcd.shape[:2]
        x_st, x_end = np.clip([x-NORM_SAMPLE_RANGE, x+NORM_SAMPLE_RANGE+1], 0, x_dim)
        y_st, y_end = np.clip([y-NORM_SAMPLE_RANGE, y+NORM_SAMPLE_RANGE+1], 0, y_dim)

        pcd = pcd[x_st:x_end, y_st:y_end]
        orig_shape = pcd.shape
        pcd = pcd.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        pcd.estimate_normals(fast_normal_computation=False,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.03, max_nn=20,
            ))

        normals = np.array(pcd.normals)
        normals = normals.reshape(orig_shape)
        normal = normals[x-x_st, y-y_st]

        if normal[2] < 0:
            normal = -normal

        return normal

    def _grasp(self):
        grasped_obj = None
        for obj in self.objects:
            grasped = self.suction_cup.grasp(obj)

            if grasped:
                grasped_obj = obj
                break
        return grasped_obj

    def step(self, action, cam='basket', pick_dir=None) -> Tuple[float, Dict]:
        currentNObject = len(self.objects)
        x, y = action // config["HalfWidth"], action % config["HalfWidth"]
        #x,y = y,x # WTF
        x, y = x * 2, y * 2
        isTerminal = False
        isUseHeuristic = False
        ret = False
        reward =0

        assert self._empty_basket_pcd is not None, "Please reset before step."
        if (not self.isEvaluation):
            self.steps += 1

        realAction = x//2*config["HalfWidth"] + y//2

        self.wait_for_stop()

        self.cams[cam].handle_explicitly()
        pcd = self.cams[cam].capture_pointcloud()
        loc = pcd[x, y]

        isTerminal = self.isTerminal()
        if (self.errorCode == DONE_EXCEEDED_MAX_ACTION or self.errorCode == DONE_FINISH):
            #print ("Never Happened here")
            reward = 0
            if (self.errorCode == DONE_FINISH):
                #print("+Reward finish return here B")
                reward += 0.5
            self.totalReward += reward
            ret = False
            returnInfo = {
                 'detect_outside': 0,
                 'pick_surface': 1,
                 'collision': -1,
                 'steps': self.steps,
                 'normal': [-1, -1, -1],  # place holder,
                'make_move': 0,  # change env
                'remaining_object': (len(self.objects)),

                 "totalPickedObject": self.countPickedObject,
                "successuflPick": ret,
                "errorCode": self.errorCode,
                "totalStep": self.steps,
                "totalActualStep": self.lastNumberAction,
                "isUseHeistic": isUseHeuristic,
                "totalReward": self.totalReward,
                 "objectOut": self.objectOut
                }

            return realAction, reward, isTerminal, returnInfo

        # pick to background
        if loc[2] - self._empty_basket_pcd[x, y, 2] < MIN_PICK_DIST2BASKET:
            reward = - 0.1
            self.totalReward += reward
            ret = False
            returnInfo = {
                'detect_outside': 0,
                'pick_surface': 1,
                'collision': -1,
                'steps': self.steps,
                'normal': [-1, -1, -1], # place holder,
                'make_move': 0,  # change env
                'remaining_object': (len(self.objects)),

                "totalPickedObject": self.countPickedObject,
                "successuflPick": ret,
                "errorCode": self.errorCode,
                "totalStep": self.steps,
                "totalActualStep": self.lastNumberAction,
                "isUseHeistic": isUseHeuristic,
                "totalReward": self.totalReward,
                 "objectOut": self.objectOut
            }

            return realAction , reward, isTerminal,returnInfo
        #startTime = time.time()

        # calculate picking orientation
        if pick_dir is None:
            pick_dir = self._normal_from_pcd(pcd, x, y)
        else:
            pick_dir = pick_dir / np.linalg.norm(pick_dir)

        if pick_dir[2] < 0:
            pick_dir = -pick_dir

        yr = np.arccos(pick_dir[2] / (np.linalg.norm([pick_dir[0], pick_dir[2]]) + 1e-6 )) / np.pi * 180
        if pick_dir[0] < 0:
            yr = -yr

        xr = np.arccos(pick_dir[2] / (np.linalg.norm([pick_dir[1], pick_dir[2]]) + 1e-6)) / np.pi * 180
        if pick_dir[1] < 0:
            xr = -xr

        yr = clip(yr, -MAX_TILT_ANG, MAX_TILT_ANG)
        xr = -clip(xr, -MAX_TILT_ANG, MAX_TILT_ANG)
        # pick_quat = euler2quat(*np.array([90, xr, 180 + yr]) / 180 * np.pi)
        pick_euler = np.array([xr, yr, 0]) / 180 * np.pi


        if self.debug:
            self.dummy_sphere.set_position(loc)

        self._set_jpos(UR5_PICK_JPOS)

        try:
            tg_pos = loc.copy()
            tg_pos += [0, 0, 0.1]
            tg_pos[2] = min(tg_pos[2], 1)
            self._move2pose(tg_pos, teleport=True)


            #if (not self.isCrashed):
            tg_pos = loc.copy()
            tg_pos += 0.05 * pick_dir
            self._move2pose(tg_pos, pick_euler)


            #if (not self.isCrashed):
            # check collision
            if self.arm.check_arm_collision(self.crate) or self.suction_cup_body.check_collision(self.crate):
                reward = - 0.1
                self.totalReward += reward
                #self.errorCode = DONE_CRASH
                #isTerminal = True
                returnInfo = {
                    'detect_outside': 0,
                    'pick_surface': 0,
                    'collision': 1,
                    'steps': self.steps,
                    'normal': [-1, -1, -1], # place holder
                    'make_move': 0,  # change env
                    'remaining_object': (len(self.objects)),

                    "totalPickedObject": self.countPickedObject,
                    "successuflPick": ret,
                    "errorCode": self.errorCode,
                    "totalStep": self.steps,
                    "totalActualStep": self.lastNumberAction,
                    "isUseHeistic": isUseHeuristic,
                    "totalReward": self.totalReward,
                     "objectOut": self.objectOut
                }

                return realAction, reward, isTerminal, returnInfo


            #if (not self.isCrashed):
            tg_pos = loc.copy()
            tg_pos -= 0.005 * pick_dir
            grasped_obj = self._move2pose(tg_pos, pick_euler, stop_at_grasp=True)
        except Exception:
            # IK failed
            reward = -1
            self.totalReward += reward
            self.errorCode = DONE_CRASH
            isTerminal = True
            returnInfo = {
                'detect_outside': 0,
                'pick_surface': 0,
                'collision': 0,
                'steps': self.steps,
                'normal': [-1, -1, -1], # place holder
                'make_move': 0,  # change env
                'remaining_object': (len(self.objects)),

                "totalPickedObject": self.countPickedObject,
                "successuflPick": ret,
                "errorCode": self.errorCode,
                "totalStep": self.steps,
                "totalActualStep": self.lastNumberAction,
                "isUseHeistic": isUseHeuristic,
                "totalReward": self.totalReward,
                "objectOut": self.objectOut
            }

            return realAction, reward, isTerminal, returnInfo


        if grasped_obj is not None:
            tg_pos = loc.copy()
            tg_pos += 0.1 * pick_dir
            tg_pos[2] = min(tg_pos[2], 1.034)

            try:
                self._move2pose(tg_pos, pick_euler)
                # self.move2jpos(UR5_PICK_JPOS)
            except Exception:
                pass
                '''
                reward = -1
                self.totalReward +=reward
                self.errorCode = DONE_CRASH
                isTerminal = True
                returnInfo = {
                    'detect_outside': 0,
                    'pick_surface': 0,
                    'collision': 1,
                    'steps': self.steps,
                    'normal': [-1, -1, -1],  # place holder

                    "totalPickedObject": self.countPickedObject,
                    "successuflPick": ret,
                    "errorCode": self.errorCode,
                    "totalStep": self.steps,
                    "totalActualStep": self.lastNumberAction,
                    "isUseHeistic": isUseHeuristic,
                    "totalReward": self.totalReward,
                     "objectOut": self.objectOut
                }

                return realAction, reward, isTerminal, returnInfo
                '''

            self.objects.remove(grasped_obj)
            grasped_obj.remove()

        detect_outside = self._remove_outside_objects()

        if (detect_outside):
            reward += -0.5
            self.objectOut +=(currentNObject - len(self.objects))

            #if (grasped_obj is not None):
            #   self.objectOut +=1

        if grasped_obj is not None:
            reward = 0.1
            ret = True
            self.countPickedObject += 1
            self.lastNumberAction = self.steps
        else:
            reward = -0.1

        isTerminal = self.isTerminal()

        if (self.errorCode == DONE_FINISH):
            #print ("+Reward finish return here B")
            reward += 0.5
        elif (self.errorCode == DONE_EXCEEDED_MAX_ACTION):
            reward = 0

        self.totalReward += reward
        returnInfo = {
            'detect_outside': 1 if detect_outside else 0,
            'pick_surface': 0,
            'collision': 0,
            'steps': self.steps,
            'normal': list(pick_dir),
            'make_move': 1, #change env
            'remaining_object': (len(self.objects)),

            "totalPickedObject": self.countPickedObject,
            "successuflPick": ret,
            "errorCode": self.errorCode,
            "totalStep": self.steps,
            "totalActualStep": self.lastNumberAction,
            "isUseHeistic": isUseHeuristic,
            "totalReward": self.totalReward,
            "objectOut": self.objectOut
            }

        return realAction , reward, isTerminal,returnInfo

    def moveToCapturePose(self):
        self._set_jpos(UR5_PICK_JPOS)
        self.pr.step()


if __name__ == '__main__':
    env = SimSuctionEnv((320, 320), headless=True)

    for _ in tqdm(itertools.count()):
        if env.done():
            env.reset()
        depth, rgb = env.take_obs()
        x, y = np.random.randint([70, 40], [240, 260])
        env.step(x, y)
