
##################
## GET ARGUMENTS
##################
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data-configuration", required=True,
	help="name of the configuration file related to the dataset to be trained. If you haven't a configuration file yet, go to the Configuration folder ")
ap.add_argument("-m", "--model-configuration", required=True,
	help="name of the model configuration file related to the model to generate. If you haven't a configuration file yet, go to the Configuration folder ")
ap.add_argument("-b", "--base-model-configuration", required=True,
	help="name of the base model configuration file related to the base model to use. If you haven't a configuration file yet, go to the Configuration folder ")
args = vars(ap.parse_args())


##################
## EXECUTE TRAIN
##################
import sys
import os

#Previous implementations were made to be executed in the src folder so let's put our execution path there
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'_src'))

from utils import generateconfig as gc
from train_utils.ssd_train import train_VOC
train_VOC(gc.get_data_config_from_name(args['data_configuration']),
			gc.get_model_config_from_name(args['model_configuration']),
			gc.get_base_model_config_from_name(args['base_model_configuration']))
