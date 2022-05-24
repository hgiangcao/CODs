import sys
sys.path.insert(1, '..')

from ITRIP.Configuration import config
from DON_Training.training import DONTrainer

DON = DONTrainer(config = config)

#DON.run(pretrained="trained_models/RGBD_8_Multi_2/DON_20400")
print ("Train DON. Mode:",config["inputMode"], "Use models:", config["model"])
DON.run()