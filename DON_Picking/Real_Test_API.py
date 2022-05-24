import numpy as np
import torch
from torchvision import transforms

import sys
sys.path.insert(1, '..')

from ITRIP.real.UR3_robot import UR3Robot, DEF_J_POS
from DON_Picking.DON_Picking_API import DON_Picking_API


class DefaultObsTransformer:
    """
    Converts depth and rgb to tensor. Clips Depth between 0 and 1.2
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

        self.rgb_transform = transforms.Compose([])

        self.depth_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.2)),
        ])

    def transform(self, depth, rgb):

        if not isinstance(depth, list) and not isinstance(
                depth, np.ndarray) and not isinstance(depth, torch.Tensor):
            depth = [depth]

        if not isinstance(rgb, list) and not isinstance(
                rgb, np.ndarray) and not isinstance(depth, torch.Tensor):
            rgb = [rgb]

        to_tensor = lambda arr: [
            a if isinstance(a, torch.Tensor) else self.to_tensor(a)
            for a in arr
        ]
        depth = to_tensor(depth)
        rgb = to_tensor(rgb)

        depth = [d.unsqueeze(0) if d.dim() < 3 else d for d in depth]
        rgb = [
            r.permute(2, 0, 1) / 255. if r.shape[0] != 3 else r for r in rgb
        ]

        depth = torch.stack([self.depth_transform(d) for d in depth]).float()
        rgb = torch.stack([self.rgb_transform(r) for r in rgb]).float()

        return depth, rgb


class RealTester:
    def __init__(self, obs_transformer=None) -> None:
        self.model = DON_Picking_API()
        if obs_transformer is None:
            self.obs_transformer = DefaultObsTransformer()
        else:
            self.obs_transformer = obs_transformer
        self.robot = UR3Robot()
        
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        
        print ("Done setup real test")

    def _calc_wd_pts(self, pts, cam2wd):
        # bring to world coordinate
        pts_h = np.concatenate([pts, np.ones((*pts.shape[:2], 1))], axis=2)

        wd_pts_h = pts_h.reshape((-1, 4))
        wd_pts_h = np.array(wd_pts_h @ cam2wd.T)
        wd_pts_h = wd_pts_h.reshape((pts_h.shape))
        wd_pts = wd_pts_h[:, :, :3]

        del wd_pts_h, pts_h

        return wd_pts

    def _calc_mask(self, rgb, wd_pts, verbose=False):
        color_norm = np.linalg.norm(rgb, axis=-1)
        color_norm_mask = color_norm < 50
        # color_norm_mask = color_norm < np.percentile(color_norm, 15)

        z_mask = wd_pts[:, :, -1] * color_norm_mask

        mask = color_norm_mask.copy()
        mask[z_mask > 0.08] = False

        if verbose:
            return color_norm, color_norm_mask, mask
        else:
            return mask

    def _get_best_action(self, rgb, depth, wd_pts):
        # calc mask
        mask = self._calc_mask(rgb, wd_pts)
        mask = 1 - mask
       
        filterMask = np.zeros((480,640))
        filterMask[120:310, 170:450] = 1

        filterMask2 = np.zeros((480,640))

        #filter to take mask by thred-depth
        for i in range (5,480-5):
            for j in range (5,640-5):
                if (depth[i,j] < 0.347):
                    filterMask2[i,j]=1


        mask = mask*filterMask*filterMask2
        tempMask = mask.copy()

        # filter mask 2. clean the mask, clean small area
        for i in range (5,480-5):
            for j in range (5,640-5):
                if (np.sum(tempMask[i-5:i+5, j-5:j+5]) < 80):
                    mask[i,j]=0

        for i in range (10,480-10):
            for j in range (10,640-10):
                meanDepth = np.mean(depth[i-10:i+10, j-10:j+10])
                minDepth = np.min(depth[i-10:i+10, j-10:j+10])
                maxDepth = np.max(depth[i-10:i+10, j-10:j+10])

                if (abs(minDepth-maxDepth) > 0.01):
                    mask[i,j]=0
                if (abs(minDepth-meanDepth) > 0.005):
                    mask[i,j]=0
                if (abs(minDepth-meanDepth) > 0.005):
                    mask[i,j]=0
                

        # calc prediction
        depth_t, rgb_t = self.obs_transformer.transform(depth[None], rgb[None])

        #print (np.sum(mask))

        x,y,des,policy = self.model.getAction(rgb_t,depth_t,mask,addData = 0)
        #print (x,y)
        #print ("mask",mask[x,y])

        #x+=10
        #y+=10
      
        return x, y,mask,des,policy

    def step(self):
        # take obs
        self.robot.move2jpos(DEF_J_POS)
        rgb, depth, pts, cam2wd = self.robot.capture_obs()

        # calc wd_pts
        wd_pts = self._calc_wd_pts(pts, cam2wd)

        # calculate pocking location
        x, y,mask,des,policy = self._get_best_action(rgb, depth, wd_pts)

        #print(x, y)

        # pick
        # if_succ = self.robot.pick_at_obs(x, y, pts, cam2wd)

        # return if_succ
        return rgb, depth,mask, pts, cam2wd,x,y,des,policy

    def pick(self,rgb, depth, pts, cam2wd,x,y):
        if_succ = self.robot.pick_at_obs(x, y, pts, cam2wd)
        return if_succ



if __name__ == '__main__':
    real_tester = RealTester()

    real_tester.step()
