from operator import xor
from typing import Dict, Tuple
from pyrep import PyRep
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.arms.arm import Arm
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
from ITRIP.assets import  UR5_GRASPNET_SCENE_RANDOM_FILE_PATH,UR5_SCENE_GRASPNET_PICKING_FILE_PATH, UR5_SCENE_FILE_PATH,UR5_SCENE_RANDOM_FILE_PATH,UR5_SCENE_ITRI_PICKING_FILE_PATH,UR3_LAB_OBJ_SCENE_FILE_PATH,UR5_LAB_OBJ_SCENE_FILE_PATH
from ITRIP.utils import center_scaling,decode_bin, encode_bin,\
    m_euler2quat as euler2quat, m_quat2euler as quat2euler, wxyz2xyzw, xyzw2wxyz
from ITRIP.objects_new import ObjectLoader,TextureRandomizer,load_graspnet_models
from ITRIP.SteroidSuctionCup import SteroidBaxterSuctionCup as BaxterSuctionCup
from ITRIP.status_code import StatusCode

from pyrep.objects import Shape
from pyrep.const import  *
from pyrep.textures.texture import Texture
from pyrep.objects.dummy import Dummy
from ITRIP.Configuration import *
from pyrep.const import ConfigurationPathAlgorithms as Algos

UR5_PICK_JPOS = [0, 0, -np.pi / 2, 0, np.pi / 2, 0]
UR5_IDLE_JPOS = [np.pi / 2, 0, -np.pi / 2, 0, np.pi / 2, 0]
NORM_SAMPLE_RANGE = 20
MAX_TILT_ANG = 60
IK_ITERS = 100
MIN_PICK_DIST2BASKET = 0.002
MAX_STOP_WAIT = 30
MAX_CONTROL_DIFF = np.pi / 180 * 5
CAM_RAND_POS_RANGE = 0.025
CAM_RAND_ROT_RANGE_DEGS = [5, 5, 5]


class InAccurateControlException(Exception):
    pass


class IKException(Exception):
    pass


class SimSuctionEnv(SuctionBaseEnv):
    def __init__(
        self,
        envIdx = 0,
        img_res=(256,256),
        headless=False,
        debug=False,
        exclude=[],
        random_texture_dir="texture",
        load_thirteen= True,
        load_lab_obj=False,
        load_ycb= False,
        load_graspnet= False,
        isEvaluation = False
    ) -> None:
        super().__init__(img_res, None)
        self.debug = debug
        np.random.seed((int)(time.time()) + envIdx)

        self.pr = PyRep()
        '''
        if (load_graspnet ):
            self.pr.launch(UR5_GRASPNET_SCENE_RANDOM_FILE_PATH, headless=headless) #for generating CODs
        elif (load_lab_obj):
            self.pr.launch(UR5_LAB_OBJ_SCENE_FILE_PATH, headless=headless)
            print ("Using LAB Objects with UR5")
        
        elif (load_thirteen):
            self.pr.launch(UR5_SCENE_ITRI_PICKING_FILE_PATH, headless=headless) # for picking
            print ("Using ITRI object with UR5 for picking")
        '''
        #UR5_SCENE_GRASPNET_PICKING_FILE_PATH
        #self.pr.launch(UR5_SCENE_GRASPNET_PICKING_FILE_PATH, headless=headless)  # for picking GraspNet
        self.pr.launch(UR5_SCENE_ITRI_PICKING_FILE_PATH, headless=headless)  # for picking ITRI
        #self.pr.launch(UR5_GRASPNET_SCENE_RANDOM_FILE_PATH, headless=headless)  # for generating CODs

        if load_graspnet:
            if(not isEvaluation):
                load_graspnet_models(self.pr, split= "train")#test_novel")
            else :
                load_graspnet_models(self.pr,split="train")


        self.pr.start()

        # set up control for robot
        self.arm = UR5()
        #self.arm = Arm(0, 'UR3', 6, max_velocity=0.2, max_acceleration=0.5, max_jerk=100)
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
        self._ini_suction_cup_config = self.suction_cup.get_configuration_tree(
        )

        # set up sensors
        # self.hand_proxi_sensor = ProximitySensor('Hand_proximity')
        self.hand_cam = VisionSensor('Hand_cam')
        self.basket_cam = VisionSensor('Basket_cam')


        self.hand_cam.set_resolution(img_res)
        self.basket_cam.set_resolution(img_res)

        self.objects = []
        self.cams = {
            'hand': self.hand_cam,
            'basket': self.basket_cam,
        }

        self.orig_cam_poses = {
            'hand': self.hand_cam.get_pose(),
            'basket': self.basket_cam.get_pose(),
        }

        try:
            self.dummy_origin = Dummy("Dummy_Origin")
            self.plane_background = Shape("Plane_Background")
            self.dummy_pos = np.array(self.dummy_origin.get_position())
        except:
            pass


        # set up object loading
        self.obj_loader = ObjectLoader(exclude=exclude,
                                       load_thirteen=load_thirteen,
                                       load_ycb=load_ycb,
                                       load_graspnet=load_graspnet,
                                       load_lab_obj=load_lab_obj)

        # set up texture randomization
        self.tex_randomizer = None
        if random_texture_dir is not None:
            self.tex_randomizer = TextureRandomizer(self.pr,
                                                    random_texture_dir)

        # set up crate
        #crate = Shape('crate_visible') #for old Crate
        crate = Shape('crate') #for new Crate
        crate.reorient_bounding_box()
        self.crate = crate
        self.crate.set_renderable(False)
        self.arm.set_renderable(False)

        min_x, max_x, min_y, max_y, min_z, max_z = crate.get_bounding_box()
        self.sample_max_dist = max(max_x - min_x, max_y - min_y,
                                   max_z - min_z) / 2

        min_z = max_z
        max_z += 0.15

        #for generate COD data
        #min_x, max_x = center_scaling(min_x, max_x, 0.7)
        #min_y, max_y = center_scaling(min_y, max_y, 1.2)

        #for picking
        min_x, max_x = center_scaling(min_x, max_x, 1)
        min_y, max_y = center_scaling(min_y, max_y, 1)
        # min_z, max_z = center_scaling(min_z, max_z, 0.9)

        self.crate_pos = np.array(crate.get_position())
        self.obj_rand_lowers = np.array([min_x, min_y, min_z]) + self.crate_pos
        self.obj_rand_highers = np.array([max_x, max_y, max_z
                                          ]) + self.crate_pos
        print (self.obj_rand_lowers, self.obj_rand_highers)
        # self.obj_rand_lowers = np.array([min_x, min_y, min_z])
        # self.obj_rand_highers = np.array([max_x, max_y, max_z])
        self._crate_bbox = np.array(crate.get_bounding_box()).reshape(
            (3, 2)).T + self.crate_pos
        self._crate_bbox = self._crate_bbox.T.flatten()
        self._crate_bbox[-1] += 0.3

        self._empty_basket_pcd = None

        self.steps = 0

        # debug stuff
        if self.debug:
            self.dummy_sphere = Shape.create(type=PrimitiveShape.SPHERE,
                                             color=[1, 0, 0],
                                             size=[0.05] * 3,
                                             respondable=False,
                                             static=True)

        self.successPick = None

    def stop(self):
        self.pr.stop()
        self.pr.shutdown()

    def changeRandomBackground(self):
        r,g,b = random.randint(0,255)/255.0, random.randint(0,255)/255.0, random.randint(0,255)/255.0
        self.plane_background.set_color([r,g,b])

    # randomize textures
    def changeRandomTexture(self):
        for obj in self.objects:
            self.tex_randomizer.randomize(obj)

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
    def wait (self):
        for _ in range (50):
            self.pr.step()
    def reset(self,
              num_objs=5,
              objectID=None,
              exclude=None,
              randomize_cam_pose=False,
              randomize_tex=True,
              listObject = None,
              randomize_tex_prob=0.5) -> Tuple[int, Dict]:

        if (self.successPick is not None):
            self.successPick.remove()

        self.successPick = None
        self.listID = []

        self.nObject = num_objs
        self._reset_arm()
        self.steps = 0

        self.errorCode = StatusCode.DONE_NONE
        self.totalReward = 0
        self.countPickedObject = 0
        self.objectOut = 0
        self.lastNumberAction = 0

        self.MAX_ATTEMP_ACTION = num_objs * 2
        #print (self.MAX_ATTEMP_ACTION)

        for obj in self.objects:
            obj.remove()

        # camera pose randomization
        [self.reset_cam_pose(cam_str) for cam_str in self.cams.keys()]
        if randomize_cam_pose:
            [self.random_cam(cam_str) for cam_str in self.cams.keys()]

        self.pr.step()
        self.basket_cam.handle_explicitly()
        self._empty_basket_pcd = self.basket_cam.capture_pointcloud()

        self.objects = []

        # random place objects
        if (listObject is not None):
            num_objs = len(listObject)

        for objth in np.arange(num_objs):
            if (listObject is not None):
                objectID = listObject[objth]

            if (objectID is None):
                obj,objID = self.obj_loader.get_random_obj(exclude=exclude)
            else:
                obj, objID = self.obj_loader.get_random_obj(exclude=exclude,objectID=objectID)


            self.listID.append(objID)

            # random pose
            rot, pos = self._sample_random_obj_pose(objth)
            #pos = np.array([0,0,0.1 ]) + self.dummy_pos#place at the center
            obj.set_orientation(rot)
            obj.set_position(pos)

            self.pr.step()

            self.objects.append(obj)

        for i in range (5):
            self.wait()
            self.wait_for_stop()
        '''
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
        '''

        # if (self.isEvaluation):
        if (len(self.objects) != num_objs):
            self.reset(num_objs=num_objs)

        # only once for training
        # self.changeRandomTexture()
        self.wait()
        self.wait_for_stop()

        return len(self.objects)

    def freezeObject(self):
        for obj in self.objects:
            obj.set_dynamic(False)
            obj.set_respondable(False)

        self.wait()

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

    def _sample_random_obj_pose(self,objth = -1):
        rot = np.random.uniform(0, 2 * np.pi, 3)

        pos = np.random.uniform(self.obj_rand_lowers, self.obj_rand_highers)

        if  (objth != -1 and objth < 9):
            px = self.obj_rand_lowers[0] + (self.obj_rand_highers[0]-self.obj_rand_lowers[0])/3*(objth//3  + random.uniform(0.4, 0.6))
            py = self.obj_rand_lowers[1] + (self.obj_rand_highers[1] - self.obj_rand_lowers[1])/3*(objth % 3 + random.uniform(0.4, 0.6))
            pos[0] = px
            pos[1] = py

        #print (pos)
        return rot, pos

    def _set_jpos(self, jpos, robust=True):
        self.arm.set_joint_positions(jpos, disable_dynamics=True)
        self.arm.set_joint_target_positions(jpos)

        if robust:
            self._move2jpos(jpos)

    def _take_obs(self, cam: VisionSensor):
        cam.handle_explicitly()  # explicit handling for faster simulation

        depth = cam.capture_depth(in_meters=True)
        rgb = cam.capture_rgb()

        # make into 255 int for memory efficiency
        # rgb = rgb * 255
        # rgb = rgb.astype(np.uint8)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

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

        if (np.max(valid_actions_decode) == 0):
            valid_actions_decode[config["HalfWidth"], config["HalfWidth"]] = 1

        valid_actions_decode = valid_actions_decode.reshape(config["W"], config["W"], 1)

        return depth, rgb, valid_actions_decode

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

    def _move2jpos(self, pos, stop_at_grasp=False):
        tg_pos = np.array(pos)

        self.arm.set_joint_target_positions(tg_pos)

        idle_cnt = 0
        max_cnt = 0
        last_jpos = np.array(self.arm.get_joint_positions())
        while np.any(
                np.abs(last_jpos -
                       tg_pos) > 0.0001) and idle_cnt < 10 and max_cnt < 30:
            self.pr.step()

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
    def _move2pose(self,
                   pos,
                   euler=np.array([0, 0, 0]),
                   teleport=False,
                   stop_at_grasp=False):
        assert not (xor(stop_at_grasp, stop_at_grasp) and
                    (teleport
                     or stop_at_grasp)), "Only teleport or stop_at_grasp"

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
        def_quat = np.quaternion(
            *xyzw2wxyz(euler2quat(*np.array([0, 0, np.pi]))))
        m_quat = np.quaternion(*xyzw2wxyz(euler2quat(*euler)))
        for i in range(IK_ITERS + 1):
            try:
                t_quat = quaternion.slerp(m_quat, def_quat, 0, 1,
                                          i / IK_ITERS).components
                t_quat = wxyz2xyzw(t_quat)
                tg_jpos = self.arm.solve_ik_via_jacobian(pos,
                                                         quaternion=t_quat)
                break
            except Exception as e:
                pass

        if tg_jpos is None:
            try:
                tg_jpos = self.arm.solve_ik_via_sampling(pos, euler=euler)[0]
            except Exception:
                pass

        if tg_jpos is None:
            try:
                tg_jpos = self.arm.solve_ik_via_jacobian(pos, euler=def_euler)
            except Exception as e:
                raise IKException(e)

        grasped_obj = None
        if teleport:
            self._set_jpos(tg_jpos)
        else:
            grasped_obj = self._move2jpos(tg_jpos, stop_at_grasp=stop_at_grasp)

            if grasped_obj is None:
                grasped_obj = self._grasp()

        # check at target
        control_diff = np.max(
            np.abs(
                np.array(self.arm.get_joint_positions()) - np.array(tg_jpos)))
        if control_diff > MAX_CONTROL_DIFF and grasped_obj is None:
            raise InAccurateControlException(control_diff)

        return grasped_obj
    '''

    def _move2pose(self,
                   pos,
                   euler=np.array([0, 0, 0]),
                   teleport=False,
                   stop_at_grasp=False):
        assert not (xor(stop_at_grasp, stop_at_grasp) and
                    (teleport
                     or stop_at_grasp)), "Only teleport or stop_at_grasp"

        path = None

        # slerping to solve path
        def_quat = np.quaternion(
            *xyzw2wxyz(euler2quat(*np.array([0, 0, -np.pi / 2]))))
        m_quat = np.quaternion(*xyzw2wxyz(euler2quat(*euler)))
        for i in range(IK_ITERS + 1):
            try:
                t_quat = quaternion.slerp(m_quat, def_quat, 0, 1,
                                          i / IK_ITERS).components
                t_quat = wxyz2xyzw(t_quat)
                path = self.arm.get_linear_path(pos,
                                                quaternion=t_quat,
                                                ignore_collisions=True,
                                                steps=5,
                                                #  algorithm=Algos.RRTstar,
                                                )
                break
            except Exception as e:
                pass

        if path is None:
            try:
                path = self.arm.get_path(pos,
                                         quaternion=def_quat,
                                         ignore_collisions=True,
                                         algorithm=Algos.RRTstar
                                         )
            except Exception as e:
                raise IKException()

        tg_jpos = path[-1]._path_points

        # move
        if teleport:
            self._set_jpos(tg_jpos)
        else:
            done = False
            while not done:
                done = path.step()
                self.pr.step()

                if stop_at_grasp:
                    obj = self._grasp()

                    if obj is not None:
                        return obj

        # check at target
        control_pos_diff = np.max(
            np.abs(self.arm.get_tip().get_position() - pos))

        # not checking gamma, along suction cup axis
        # control_euler_diff = np.max(
        #     np.abs(self.arm.get_tip().get_orientation()[:2] - euler[:2]))
        # if control_pos_diff > 0.005 or control_euler_diff > np.pi / 180 * 5:
        if control_pos_diff > 0.005:
            raise InAccurateControlException(
                F"control_pos_diff: {control_pos_diff}"
            )
    def _reset_arm(self):
        self.pr.set_configuration_tree(self._ini_arm_config)
        self.pr.set_configuration_tree(self._ini_suction_cup_config)
        self._set_jpos(UR5_IDLE_JPOS)
        self.suction_cup.release()

    def take_obs(self,
                 hand_obs=False,
                 basket_obs=True) -> Tuple[np.array, np.array]:
        self._reset_arm()
        self.pr.step()

        obs = self._take_selected_obs(hand_obs, basket_obs)

        if (self.successPick is not None):
            self.wait()
            self.successPick.remove()

        return obs

    def done(self) -> bool:
        return len(self.objects) == 0

    def _normal_from_pcd(self, pcd, x, y):
        x_dim, y_dim = pcd.shape[:2]
        x_st, x_end = np.clip(
            [x - NORM_SAMPLE_RANGE, x + NORM_SAMPLE_RANGE + 1], 0, x_dim)
        y_st, y_end = np.clip(
            [y - NORM_SAMPLE_RANGE, y + NORM_SAMPLE_RANGE + 1], 0, y_dim)

        pcd = pcd[x_st:x_end, y_st:y_end]
        orig_shape = pcd.shape
        pcd = pcd.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        pcd.estimate_normals(fast_normal_computation=False,
                             search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                 radius=0.03,
                                 max_nn=20,
                             ))

        normals = np.array(pcd.normals)
        normals = normals.reshape(orig_shape)
        normal = normals[x - x_st, y - y_st]

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

    def reset_cam_pose(self, cam_name: str):
        cam = self.cams[cam_name]
        orig_pose = self.orig_cam_poses[cam_name]
        cam.set_pose(orig_pose)

    def random_cam(self, cam_name: str):
        cam = self.cams[cam_name]

        pos_rand = np.random.uniform(-CAM_RAND_POS_RANGE, CAM_RAND_POS_RANGE,
                                     3)
        euler_rand = np.random.uniform(-np.array(CAM_RAND_ROT_RANGE_DEGS), \
            CAM_RAND_ROT_RANGE_DEGS) / 180 * np.pi

        self.reset_cam_pose(cam_name)
        cam.rotate(euler_rand)
        cam.set_position(cam.get_position() + pos_rand)

    def isTerminal(self):
        # grab all objects
        if (self.done()):
            self.errorCode = StatusCode.DONE_FINISH
            return True

        if (self.steps > self.MAX_ATTEMP_ACTION):
            self.errorCode = StatusCode.DONE_EXCEEDED_MAX_ACTION
            return True

        return False

    def step(self, action, cam='basket', pick_dir=None) -> Tuple[float, Dict]:
        # pick_dir = [0,0,1]
        ret = 0
        self.successPick = None
        self.errorCode = StatusCode.DONE_NONE
        isTerminal = False
        realAction = 0
        currentNObject = len(self.objects)
        x, y = action // config["HalfWidth"], action % config["HalfWidth"]
        # x,y = y,x # WTF
        x, y = x * 2, y * 2
        isTerminal = False
        isUseHeuristic = False
        ret = False
        reward = -0.1
        self.steps += 1

        assert self._empty_basket_pcd is not None, "Please reset before step."

        # coff = (self.steps+1)//2

        self.wait_for_stop()

        self.cams[cam].handle_explicitly()
        pcd = self.cams[cam].capture_pointcloud()
        loc = pcd[x, y]
        # max exxed
        isTerminal = self.isTerminal()
        if (self.errorCode == StatusCode.DONE_EXCEEDED_MAX_ACTION or self.errorCode == StatusCode.DONE_FINISH):
            if (self.errorCode == StatusCode.DONE_EXCEEDED_MAX_ACTION):
                reward += -1
            self.totalReward += reward
            # isTerminal = True
            returnInfo = {
                'steps': self.steps,
                'num_obj_left': len(self.objects),
                'detect_outside': 0,
                'errorCode': self.errorCode,
                'make_move': 0,

                "totalPickedObject": self.countPickedObject,
                "totalActualStep": self.lastNumberAction,
                "totalReward": self.totalReward,
                "objectOut": self.objectOut,
                'remaining_object': (len(self.objects)),
                "ret": ret,
                "nObject": self.nObject
            }

            return realAction, reward, isTerminal, returnInfo

        if loc[2] - self._empty_basket_pcd[x, y, 2] < MIN_PICK_DIST2BASKET:
            reward += -0.1
            self.totalReward += reward
            self.errorCode = StatusCode.PICK_ON_SURFACE
            returnInfo = {
                'steps': self.steps,
                'num_obj_left': len(self.objects),
                'detect_outside': 0,
                'errorCode': self.errorCode,
                'make_move': 0,

                "totalPickedObject": self.countPickedObject,
                "totalActualStep": self.lastNumberAction,
                "totalReward": self.totalReward,
                "objectOut": self.objectOut,
                'remaining_object': (len(self.objects)),
                "ret": ret,
                "nObject": self.nObject
            }
            return realAction, reward, isTerminal, returnInfo

        # calculate picking orientation
        if pick_dir is None:
            pick_dir = self._normal_from_pcd(pcd, x, y)
        else:
            pick_dir = pick_dir / np.linalg.norm(pick_dir)

        if pick_dir[2] < 0:
            pick_dir = -pick_dir

        yr = np.arccos(
            pick_dir[2] /
            (np.linalg.norm([pick_dir[0], pick_dir[2]]) + 1e-6)) / np.pi * 180
        if pick_dir[0] < 0:
            yr = -yr

        xr = np.arccos(
            pick_dir[2] /
            (np.linalg.norm([pick_dir[1], pick_dir[2]]) + 1e-6)) / np.pi * 180
        if pick_dir[1] < 0:
            xr = -xr

        yr = np.clip(yr, -MAX_TILT_ANG, MAX_TILT_ANG)
        xr = -np.clip(xr, -MAX_TILT_ANG, MAX_TILT_ANG)
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

            tg_pos = loc.copy()
            tg_pos += 0.05 * pick_dir
            self._move2pose(tg_pos, pick_euler)

            # check collision
            if self.arm.check_arm_collision(
                    self.crate) or self.suction_cup_body.check_collision(
                self.crate):
                reward += -0.1
                self.totalReward += reward
                self.errorCode = StatusCode.COLLISION
                returnInfo = {
                    'steps': self.steps,
                    'num_obj_left': len(self.objects),
                    'detect_outside': 0,
                    'errorCode': self.errorCode,
                    'make_move': 0,

                    "totalPickedObject": self.countPickedObject,
                    "totalActualStep": self.lastNumberAction,
                    "totalReward": self.totalReward,
                    "objectOut": self.objectOut,
                    'remaining_object': (len(self.objects)),
                    "ret": ret,
                    "nObject": self.nObject
                }
                return realAction, reward, isTerminal, returnInfo

            tg_pos = loc.copy()
            tg_pos -= 0.005 * pick_dir

            grasped_obj = self._move2pose(tg_pos,
                                          pick_euler,
                                          stop_at_grasp=True)
        except IKException:
            # IK failed
            reward += -1
            self.totalReward += reward
            self.errorCode = StatusCode.IK_FAILURE
            # if (not isEvaluation):
            isTerminal = True
            returnInfo = {
                'steps': self.steps,
                'num_obj_left': len(self.objects),
                'detect_outside': 0,
                'errorCode': self.errorCode,
                'make_move': 0,

                "totalPickedObject": self.countPickedObject,
                "totalActualStep": self.lastNumberAction,
                "totalReward": self.totalReward,
                "objectOut": self.objectOut,
                'remaining_object': (len(self.objects)),
                "ret": ret,
                "nObject": self.nObject
            }

            return realAction, reward, isTerminal, returnInfo
        except InAccurateControlException:
            # Control failure
            # isTerminal = True
            reward += -0.05
            self.totalReward += reward
            self.errorCode = StatusCode.CONTROL_FAILURE
            # isTerminal = True

            returnInfo = {
                'steps': self.steps,
                'num_obj_left': len(self.objects),
                'detect_outside': 0,
                'errorCode': self.errorCode,
                'make_move': 1,

                "totalPickedObject": self.countPickedObject,
                "totalActualStep": self.lastNumberAction,
                "totalReward": self.totalReward,
                "objectOut": self.objectOut,
                'remaining_object': (len(self.objects)),
                "ret": ret,
                "nObject": self.nObject
            }

            return realAction, reward, isTerminal, returnInfo

        if grasped_obj is not None:
            tg_pos = loc.copy()
            tg_pos += 0.1 * pick_dir
            tg_pos[2] = min(tg_pos[2], 1.034)

            '''
            #MOVE UP AFTER PICK OBJECT.
            #COMMENT to ignore the moving up action after picking
            try:
                self._move2pose(tg_pos, pick_euler)
                # self.move2jpos(UR5_PICK_JPOS)
            except Exception:
                # safe fail, failed at retreating, not matter
                pass
            '''

            if grasped_obj is not None:
                reward += 0.1
                ret = True
                self.countPickedObject += 1
                self.lastNumberAction = self.steps
                ret = 1
                self.successPick = grasped_obj
            else:
                reward += -0.1

            self.objects.remove(grasped_obj)
            #grasped_obj.remove()
            # self.suction_cup.release()
            # self.wait()
            # grasped_obj.remove()

        detect_outside = self._remove_outside_objects()

        if (detect_outside):
            reward += -0.5
            self.objectOut += (currentNObject - len(self.objects))

        self.errorCode = StatusCode.DONE_FAIL if grasped_obj is None else StatusCode.DONE_SUCC

        isTerminal = self.isTerminal()
        # if (self.errorCode == StatusCode.DONE_FINISH ):
        #    reward += 0.5

        self.totalReward += reward

        returnInfo = {
            'steps': self.steps,
            'num_obj_left': len(self.objects),
            'detect_outside': detect_outside,
            'errorCode': self.errorCode,
            'make_move': 1,

            "totalPickedObject": self.countPickedObject,
            "totalActualStep": self.lastNumberAction,
            "totalReward": self.totalReward,
            "objectOut": self.objectOut,
            'remaining_object': (len(self.objects)),
            "ret": ret,
            "nObject": self.nObject
        }

        return realAction, reward, isTerminal, returnInfo

    def getRemainingObject(self):
        return len(self.objects)


if __name__ == '__main__':
    env = SimSuctionEnv((320, 320), headless=True)

    for _ in tqdm(itertools.count()):
        if env.done():
            env.reset()
        depth, rgb = env.take_obs()
        x, y = np.random.randint([70, 40], [240, 260])
        env.step(x, y)
