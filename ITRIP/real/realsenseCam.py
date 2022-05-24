import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from skimage.restoration import inpaint


class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        profile = self.pipeline.start(config)
        device = profile.get_device()

        depth_sensor = device.first_depth_sensor()
        color_sensor = device.first_color_sensor()

        # configure
        depth_sensor.set_option(rs.option.emitter_always_on, True)

        # get filters
        self.align = rs.align(rs.stream.color)
        self.depth_filters = [
            # rs.decimation_filter(2),
            rs.hdr_merge(),
            rs.threshold_filter(0.2, 1.2),
            rs.disparity_transform(True),
            rs.spatial_filter(0.3, 18, 2, 0),
            rs.temporal_filter(0.2, 80, 7),
            rs.hole_filling_filter(2),
            rs.disparity_transform(False),
        ]

        # point cloud stuff
        self.pc = rs.pointcloud()

    def capture_obs(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        for f in self.depth_filters:
            depth_frame = f.process(depth_frame)

        # get depth and rgb
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float) / 1000
        rgb = np.asanyarray(color_frame.get_data())

        # calc pcd
        points = self.pc.calculate(depth_frame)
        pts = np.asanyarray(points.get_vertices(3))

        # if inpaint_depth:
        #     orig_depth = depth.copy()

        #     # down-scale and inpaint
        #     down_depth = cv2.resize(depth, tuple(np.array(depth.shape)[::-1] // self.depth_inpaint_down_factor), interpolation=cv2.INTER_NEAREST)
        #     mask = np.zeros_like(down_depth)
        #     mask[down_depth == 0] = 1
        #     down_depth_inpaint = inpaint.inpaint_biharmonic(down_depth, mask)

        #     # up-sacle to original
        #     depth = cv2.resize(down_depth_inpaint, depth.shape[::-1], interpolation=cv2.INTER_NEAREST)

        #     # put original valid depth back
        #     depth[orig_depth!=0] = orig_depth[orig_depth!=0]

        return rgb, depth, pts

    def calc_pcd(self, pts, rgb=None, extrinsic=np.eye(4), get_pcd=False):
        orig_shape = pts.shape[:2]

        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pts.reshape(-1, 3)))

        if rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255)

        pcd.transform(extrinsic)

        # compute normal
        pcd.estimate_normals(fast_normal_computation=False,
                             search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                 radius=0.05,
                                 max_nn=50,
                             ))

        if get_pcd:
            return pcd
        else:
            normals = np.array(pcd.normals).reshape((*orig_shape, 3))
            pts = np.asarray(pcd.points).reshape((*orig_shape, 3))

            return pts, normals

    def close(self):
        self.pipeline.stop()
