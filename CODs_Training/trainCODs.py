import sys
sys.path.insert(1, '..')

from ITRIP.Configuration import config
from CODs_Training.training import CODsTrainer

CODs = CODsTrainer(config = config)

#CODs.run(pretrained="trained_models/RGBD_8_Multi_2/CODs_20400")
print ("Train CODs. Mode:",config["inputMode"], "Use models:", config["model"])
CODs.run()