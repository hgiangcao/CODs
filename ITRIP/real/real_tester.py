import numpy as np
import torch
from torchvision import transforms

from ITRIP.real.UR3_robot import UR3Robot, DEF_J_POS


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
    def __init__(self, model, obs_transformer=None) -> None:
        self.model = model
        if obs_transformer is None:
            self.obs_transformer = DefaultObsTransformer()
        else:
            self.obs_transformer = obs_transformer
        self.robot = UR3Robot()

        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

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
        mask[z_mask > 0.05] = False

        if verbose:
            return color_norm, color_norm_mask, mask
        else:
            return mask

    def _get_best_action(self, rgb, depth, wd_pts):
        # calc mask
        mask = self._calc_mask(rgb, wd_pts)

        # calc prediction
        depth_t, rgb_t = self.obs_transform.transform(depth[None], rgb[None])
        with torch.no_grad():
            pred = self.model(rgb_t.cuda(), depth_t.cuda())
        pred = pred[0].cpu().detach().numpy()

        # apply mask
        pred[mask] = float('-Inf')

        # apply basket area mask
        non_basket_area_mask = np.ones_like(pred).astype(np.bool_)
        non_basket_area_mask[40:-70, 80:-100] = False
        pred[non_basket_area_mask] = float('-Inf')

        # get best action
        x, y = np.unravel_index(pred.reshape(-1).argmax(axis=0), pred.shape)
        x, y = np.array([x, y]).T

        return x, y

    def step(self):
        # take obs
        self.robot.move2jpos(DEF_J_POS)
        rgb, depth, pts, cam2wd = self.robot.capture_obs()

        # calc wd_pts
        wd_pts = self._calc_wd_pts(pts, cam2wd)

        # calculate pocking location
        x, y = self._get_best_action(rgb, depth, wd_pts)

        print(x, y)

        # pick
        # if_succ = self.robot.pick_at_obs(x, y, pts, cam2wd)

        # return if_succ


if __name__ == '__main__':
    model = torch.jit.load(
        '/home/giang/Desktop/UNET_RAND_D_jit_model.pth').cuda()
    real_tester = RealTester(model)