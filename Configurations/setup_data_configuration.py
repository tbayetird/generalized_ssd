import sys
import os
#Previous implementations were made to be executed in the src folder so let's put our execution path there
config_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(os.path.dirname(config_directory),'_src'))
from utils import generateconfig as genc

##################
## GET ARGUMENTS
##################
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of the configuration file related to the dataset to be trained that will be generated. ")
ap.add_argument("-m", "--model_name", required=False,
	default = "example_model",
	help="name of the model that will be generated during training or that is already here. Default : example_model")
ap.add_argument("-p", "--path", required=True,
	help="path to the directory containing the dataset. The dataset should be organised as the PASCAL VOC format for now ")
ap.add_argument("-o", "--output-path", required=False,
    default=os.path.join(config_directory,'config_datas'),
    help=" path to the directory where the configuration will be generated. Default : Configurations/config_datas ")
ap.add_argument("-e", "--epochs", required=False,
    default=40,
    help=" Number of epochs that should be used for training. Ignore if no training should be done. Default : 40 ")
ap.add_argument("-b", "--batch-size", required=False,
    default=16,
    help="Size of the batch used for training. Ignore if no training should be done. Default : 16")
ap.add_argument("-t", "--test-batch-size", required=False,
    default=1,
    help="size of the batch used when testing on the test dataset. Ignore if no testing should be done. Default : 1")
args = vars(ap.parse_args())

##################
## SETUP CONFIGURATION
##################
import gc
## Generating data configuration
genc.setup_data_config(args['path'],
					args['name'],
					args['model_name'],
					args['output_path'],
					args['epochs'],
					args['batch_size'],
					args['test_batch_size']
					)
