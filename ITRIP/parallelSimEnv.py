from numpy.core.fromnumeric import reshape
import ray
import numpy as np
from typing import Dict, List, Tuple

from ITRIP.simEnv import SimSuctionEnv
from ITRIP.utils import transpose

@ray.remote(num_cpus=3, num_gpus=0.3)
class ParallelSimWorker(object):

    def __init__(self, img_res, seed=None, exclude=[], headless=True) -> None:
        if seed is None:
            seed = np.random.randint(0, 2**31)
        np.random.seed(seed)

        self.sim = SimSuctionEnv(img_res=img_res, 
                                 headless=headless, 
                                 debug=False,
                                 exclude=exclude)

    def reset(self, num_objs=1, hand_obs=False, basket_obs=True, select=True) -> int:
        if select:
            return self.sim.reset(num_objs=num_objs,
                              hand_obs=hand_obs,
                              basket_obs=basket_obs)
        else:
            return None

    @ray.method(num_returns=3)
    def take_obs(self, hand_obs=False, basket_obs=True) -> Tuple[np.array, np.array]:
        return self.sim.take_obs(hand_obs=hand_obs,
                                 basket_obs=basket_obs)

    def done(self) -> bool:
        return self.sim.done()
    
    def stop(self) -> None:
        self.sim.stop()

    @ray.method(num_returns=2)
    def step(self, x, y, cam='basket', pick_dir=None) -> Tuple[float, Dict]:
        return self.sim.step(x=x, y=y, cam=cam, pick_dir=pick_dir)


class VectorParallelSim(object):

    def __init__(self, img_res, num_env, exclude=[], headless=True) -> None:
        super().__init__()
        seeds = np.random.randint(0, 2**31, size=num_env)
        self.num_env = num_env

        self.sims = [ParallelSimWorker.remote(img_res, seed=seed, exclude=exclude, headless=headless)for seed in seeds]

    def reset(self, num_objs=1, hand_obs=False, basket_obs=True, select=None) -> List[int]:
        if select is None:
            select = [True for _ in range(self.num_env)]

        res = [sim.reset.remote(
                        num_objs=num_objs, 
                        hand_obs=hand_obs,
                        basket_obs=basket_obs,
                        select=sel)
                        for sel, sim in zip(select, self.sims)]

        return res

    def take_obs(self, hand_obs=False, basket_obs=True) -> Tuple[List[np.array], List[np.array]]:
        res = [sim.take_obs.remote(
                            hand_obs=hand_obs,
                            basket_obs=basket_obs)
                            for sim in self.sims]
        
        return transpose(res)

    def done(self) -> List[bool]:
        return [sim.done.remote() for sim in self.sims]

    def stop(self):
        [sim.stop.remote() for sim in self.sims]

    def step(self, actions, cam='basket', pick_dir=None) -> Tuple[List[float], List[Dict]]:
        if pick_dir is None:
            pick_dir = [None] * len(self.sims)

        res = [sim.step.remote(*actions[i], cam=cam, pick_dir=pick_dir[i])
                for i, sim in enumerate(self.sims)]

        return transpose(res)
