from spartan_dataset_masked import SpartanDataset
import dense_correspondence_manipulation.utils.utils as utils

utils.add_dense_correspondence_to_python_path()
import correspondence_tools.correspondence_finder as correspondence_finder
import correspondence_tools.correspondence_plotter as correspondence_plotter
from dataset.dense_correspondence_dataset_masked import ImageType

import os
import torch
import numpy as np
from matplotlib import pyplot as plt

dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset', 'composite',
                                       'Donkey.yaml')

dataset_config = utils.getDictFromYamlFilename(dataset_config_filename)
print (dataset_config)
dataset = SpartanDataset(debug=True, config=dataset_config)

match_type, image_a_rgb, image_b_rgb, \
matches_a, matches_b, masked_non_matches_a, \
masked_non_matches_a, non_masked_non_matches_a, \
non_masked_non_matches_b, blind_non_matches_a, \
blind_non_matches_b, metadata = dataset.get_single_object_within_scene_data()