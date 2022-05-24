import sys
sys.path.insert(1, '..')
from ITRIP.newSimEnv_2 import SimSuctionEnv
import torch.multiprocessing as mp
from ITRIP.Configuration import  *
import cv2
import torch

class MultiSimEnv(SimSuctionEnv):
    def __init__(self,envIdx , img_res,envName ="env", headless=False, debug=False, exclude=['object12'],isEvaluation = False) -> None:
        super().__init__(envIdx,img_res , headless, debug, exclude,isEvaluation = isEvaluation)

        self.envName = envName

    def finish(self):
        self.pr.stop()
        self.pr.shutdown()
        print (self.envName," FINISHED")


class MultipleEnvironments:
    def __init__(self, num_envs=10, headless = True,isEvaluation = False):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.isEvaluation = isEvaluation
        self.envs = [None] * num_envs

        self.numEnv = num_envs
        self.headless = headless

        for index in range(self.numEnv):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

        print("Done create Multiple environment")
        print (len( self.agent_conns),len(self.envs))




    def run(self, index):
        if (index == self.numEnv-1  and self.isEvaluation):
            headless = False # visualiz the last env
        else :
            headless = True
        self.envs[index]  = MultiSimEnv( envIdx= index, envName = "env" +str(index), img_res = (config["W"], config["W"]), headless= headless,isEvaluation=self.isEvaluation)
        self.agent_conns[index].close()
        while True:
            request, param = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(param))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset(param))
            elif request == "finish":
                self.env_conns[index].send(self.envs[index].finish())
            elif request == "getObservation":
                self.env_conns[index].send(self.envs[index].take_obs())
            else:
                raise NotImplementedError
