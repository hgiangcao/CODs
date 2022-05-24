import numpy as np
import abc

# This should be the API for environment
class SuctionBaseEnv(abc.ABC):

    def __init__(self, img_res, cam_intrinsics) -> None:
        super().__init__()

        self.img_res = img_res
        self.cam_intrinsics = cam_intrinsics

    def reset(self) -> dict:
        # reset the basket and return observation

        raise NotImplementedError


    def step(self, x, y) -> dict:
        # take action and return whether the picking succeeded

        raise NotImplementedError

    def done(self) -> bool:
        # return whether the basket is empty

        raise NotImplementedError

    def info(self) -> any:
        # return environment specific information

        raise None