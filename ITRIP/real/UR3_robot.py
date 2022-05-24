import urx
import rtde_receive
import time
import numpy as np
from transforms3d.euler import euler2mat
import math3d as m3d
import logging
import random

from ITRIP.real.realsenseCam import Camera
from ITRIP.assets import VACUUM_PICK_SCRIPT, VACUUM_RELEASE_SCRIPT, VACUUM_CHECK_SCRIPT

# DEF_J_POS = np.array([-180, -90, 90, 270, 270, -180]) / 180 * np.pi
# DEF_J_POS = np.array([-3.14157373, -1.57188589,  0.76273584,  5.52175522,  4.71246767, -3.14166075])
DEF_J_POS = np.array([-3.141573731099264, -1.6777856985675257, 1.0525007247924805, 5.337910175323486, 4.712467670440674, -3.1416967550860804])
DROP_JOPS1 = [-3.9432836214648646, -1.3187621275531214, 0.7863659858703613, 5.237644195556641, 4.711938858032227, -3.9433417955981653]
DROP_JOPS2 = [-2.624390188847677, -1.0308201948748987, 0.4356346130371094, 5.307643890380859, 4.712527751922607, -2.624509636555807]

PICKING_DISTS = [0, 0.005, 0.01]
MAX_TILT_ANG = 15
DEF_TCP = [0, 0, 0.22, -2.22, -2.22, 0]

CHECK_INTERV_SEC = 0.05

# TODO: prepare hand-eye coordination
# measued
# CAM2TCP_TRANSLATION = np.array([0.076 + 0.0175, -0.036 + 0.025, 0.1637 + 0.002])
CAM2TCP_TRANSLATION = np.array([0.076, -0.036, 0.1637 + 0.002])
CAM2TCP_ROT = euler2mat(*np.array([0, 180, -90]) / 180 * np.pi)

class UR3Robot:
    def __init__(self, host="10.5.11.70", min_valid_height=0.04):
        self.robot = urx.Robot(host, use_rt=True)
        self.rtde_receive = rtde_receive.RTDEReceiveInterface(host)

        self.cam = None
        self.robot.set_tcp(DEF_TCP)

        self._reset_cam()

        self.num_picked = 0
        self.min_valid_height = min_valid_height

    def wait_prog(self, timeout_sec=30):
        st_time = time.time()

        # wait for program to start
        while time.time() - st_time < 0.5 and not self.robot.is_program_running():
            time.sleep(CHECK_INTERV_SEC)
            pass

        while time.time() - st_time < timeout_sec:
            time.sleep(CHECK_INTERV_SEC)
            
            if not self.robot.is_program_running():
                break

    def _reset_cam(self):
        if self.cam is not None:
            self.cam.close()
            time.sleep(0.1)
        self.cam = Camera()

    def capture_obs(self, *args, **kwargs):
        cam2wd = self.get_cam2wd()

        obs = None

        for _ in range(3):
            # fails sometimes for no reason
            try:
                obs = self.cam.capture_obs(*args, **kwargs)
                break
            except Exception as e:
                logging.warn('Failed taking obs from cam. Reset camera', e)
                self._reset_cam()

        if obs is None:
            obs = self.cam.capture_obs(*args, **kwargs)

        return *obs, cam2wd
    def capture_rgbd(self):
        return self.cam.capture_obs()


    def calc_pcd(self, *args, cam2wd=None, **kwargs):
        if cam2wd is None:
            cam2wd = self.get_cam2wd()
        return self.cam.calc_pcd(*args, **kwargs, extrinsic=cam2wd)

    def move2jpos(self, jpose, acc=1, vel=0.5, timeout_sec=30, wait=True):
        self.robot.movej(jpose, acc=acc, vel=vel, wait=False)
        if wait:
            self.wait_prog(timeout_sec)

    def get_jpos(self):
        return self.robot.getj()

    def move2pose(self, pose, acc=0.5, vel=0.2, timeout_sec=30):
        self.robot.movel(pose, acc=acc, vel=vel, wait=False)
        self.wait_prog(timeout_sec)

    def get_pose(self):
        pose = self.robot.get_pose()

        return pose

    def pick(self):
        with open(VACUUM_PICK_SCRIPT, 'r') as f:
            prog = ''.join(f.readlines())
        self.robot.send_program(prog)
        self.wait_prog()

    def check_pick(self):
        with open(VACUUM_CHECK_SCRIPT, 'r') as f:
            prog = ''.join(f.readlines())
        self.robot.send_program(prog)
        self.wait_prog()

        if_pick = self.rtde_receive.getDigitalOutState(0)

        return if_pick

    def get_cam2wd(self):
        tcp2wd = self.get_pose().get_matrix()

        cam2tcp = np.eye(4)
        cam2tcp[:3, :3] = CAM2TCP_ROT
        cam2tcp[:3, -1] = CAM2TCP_TRANSLATION

        cam2wd = tcp2wd @ cam2tcp

        return cam2wd

    def release(self):
        with open(VACUUM_RELEASE_SCRIPT, 'r') as f:
            prog = ''.join(f.readlines())
        self.robot.send_program(prog)
        self.wait_prog()

        # really werid bug where we need to run pick after release
        # to actually release
        with open(VACUUM_PICK_SCRIPT, 'r') as f:
            prog = ''.join(f.readlines())
        self.robot.send_program(prog)
        self.wait_prog()

    def calc_pick_pose(self, x, y, pts, cam2wd, pick_dir=None):
        pts, normals = self.calc_pcd(pts, get_pcd=False, cam2wd=cam2wd)

        # get picking position and direction
        loc = pts[x, y]
        if pick_dir is None:
            pick_dir = normals[x, y]

        if pick_dir[2] < 0:
            pick_dir = -pick_dir

        # calc picking pose
        yr = np.arccos(pick_dir[2] / (np.linalg.norm([pick_dir[0], pick_dir[2]]) + 1e-6 )) / np.pi * 180
        if pick_dir[0] < 0:
            yr = -yr

        xr = np.arccos(pick_dir[2] / (np.linalg.norm([pick_dir[1], pick_dir[2]]) + 1e-6)) / np.pi * 180
        if pick_dir[1] < 0:
            xr = -xr

        yr = np.clip(yr, -MAX_TILT_ANG, MAX_TILT_ANG)
        xr = -np.clip(xr, -MAX_TILT_ANG, MAX_TILT_ANG)
        pick_euler = np.array([xr, yr, 0]) / 180 * np.pi

        return loc, pick_dir, pick_euler

    def reset(self):
        self.num_picked = 0

    def pick_at_obs(self, x, y, pts, cam2wd, pick_dir=None):
        # check valid
        if np.linalg.norm(pts[x, y]) < 0.01:
            return False

        # move to picking location
        self.move2jpos(DEF_J_POS, wait=False, acc=0.5, vel=0.5)

        # get picking pose
        loc, pick_dir, pick_euler = self.calc_pick_pose(x, y, pts, cam2wd, pick_dir=pick_dir)

        # wait to be at init pos
        self.wait_prog(5)

        # move above
        pick_pose = m3d.Transform()
        pick_pose.orient.set_array(euler2mat(*pick_euler))
        pick_pose.pos.set_array(loc + pick_dir * 0.02)
        self.move2pose(pick_pose, timeout_sec=5)

        # try picking at multiple level
        if_picked = False

        for into_dist in PICKING_DISTS:
            pick_pose = m3d.Transform()
            pick_pose.orient.set_array(euler2mat(*pick_euler))
            pick_pose.pos.set_array(loc - pick_dir * into_dist)
            self.move2pose(pick_pose, acc=0.1, vel=0.05)
            self.pick()

            time.sleep(0.05)
            if_picked = np.all([self.check_pick() for _ in range(3)])

            if if_picked:
                break

        # retrieve
        pick_pose = m3d.Transform()
        pick_pose.orient.set_array(euler2mat(*pick_euler))
        pick_pose.pos.set_array(loc + [0, 0, 0.02])
        self.move2pose(pick_pose, acc=0.2, vel=0.1)

        # check pick
        time.sleep(0.05)
        if_picked = np.all([self.check_pick() for _ in range(3)])
        if if_picked:
            self.move2jpos(DEF_J_POS, acc=0.5, vel=0.5)

            # double check
            if_picked = np.all([self.check_pick() for _ in range(3)])

        # drop if picked
        if if_picked:
            drop_jpos = DROP_JOPS1 if self.num_picked < 15 else DROP_JOPS2
            # drop_jpos = DROP_JOPS1 if self.num_picked % 2 == 0 else DROP_JOPS2
            self.num_picked += 1

            self.move2jpos(drop_jpos)
            dx = (random.randint(0,10)-5)/100
            dy = (random.randint(0,10)-5)/100

            currentPose = self.get_pose()
            currentPose.pos.set_array(currentPose.pos.get_array() + [dx, dy, -0.05])
            self.move2pose(currentPose, acc=0.2, vel=0.1)

            self.release()

        self.move2jpos(DEF_J_POS, acc=0.5, vel=1)

        return if_picked

    def stop(self):
        self.robot.stop()