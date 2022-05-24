from pyrep.objects.vision_sensor import VisionSensor
import numpy as np

from copy import deepcopy
from pycocotools import mask
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2quat, quat2euler


def wxyz2xyzw(quat):
    return np.array([*quat[1:], quat[0]])


def xyzw2wxyz(quat):
    return np.array([quat[3], *quat[:3]])


def m_euler2quat(*args, **kwargs):
    res = euler2quat(*args, *kwargs)

    return wxyz2xyzw(res)


def m_quat2euler(quat):
    quat = xyzw2wxyz(quat)

    return quat2euler(quat)


def encode_bin(arr: np.array):
    fortran_arr = np.asfortranarray(arr.astype(np.uint8))

    dec = mask.encode(fortran_arr)

    return dec


def pose2mat(pos, quat=None):
    if quat is None:
        quat = pos[3:]
        pos = pos[:3]

    mat = np.eye(4)
    mat[:3, :3] = quat2mat(quat)
    mat[:3, 3] = pos

    return mat

def mat2pose(mat):
    pos = mat[:3, 3]
    quat = mat2quat(mat[:3, :3])

    return pos, quat


def decode_bin(enc):
    return mask.decode(enc)


def transpose(arr):
    return list(zip(*arr))

'''
def ray_nested_get(obj):
    if isinstance(obj, ray.ObjectRef):
        o = ray.get(obj)
        del obj

        if 'copy' in dir(o):
            o = o.copy()
        else:
            o = deepcopy(o)
        return o

    if isinstance(obj, list) or isinstance(obj, tuple):
        return [ray_nested_get(o) for o in obj]

    return obj 
'''

def center_scaling(num1, num2, sclae):
    mid = (num1 + num2) / 2

    num1 = mid + (num1 - mid) * sclae
    num2 = mid + (num2 - mid) * sclae

    return num1, num2


def get_cam_intrinsics(cam: VisionSensor) -> np.array:
    fov = cam.get_perspective_angle()
    img_w, img_h = cam.get_resolution()

    f = img_w / (2 * np.tan(fov / 2 * np.pi / 180))

    intrinsics_mat = np.array([
        [f, 0 , img_w / 2],
        [0, f , img_h / 2],
        [0, 0 , 1],
    ])

    return intrinsics_mat


def depth2pcd(depth: np.array, intrin_mat: np.array) -> np.array:
    img_h, img_w = depth.shape

    X = np.arange(0, img_w)
    Y = np.arange(0, img_h)
    xv, yv = np.meshgrid(X, Y)

    pts = np.stack([xv, yv, np.ones_like(yv)]).T.reshape(-1, 3)

    pts = pts @ np.linalg.pinv(intrin_mat).T
    pts = pts / np.linalg.norm(pts, axis=1)[:, None]

    pts = pts * depth.flatten()[:, None]

    return pts


def partial_depth2pcd(depth: np.array, coord: np.array, intrin_mat: np.array) -> np.array:
    assert coord.ndim == 2
    assert depth.ndim == 1
    assert coord.shape[0] == depth.shape[0]

    pts = np.concatenate([coord, np.ones((coord.shape[0], 1))], axis=1)
    pts = pts @ np.linalg.pinv(intrin_mat).T
    pts = pts / np.linalg.norm(pts, axis=1)[:, None]
    pts = pts * depth.flatten()[:, None]

    return pts
