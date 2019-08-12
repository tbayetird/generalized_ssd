import sys
import os
#Previous implementations were made to be executed in the src folder so let's put our execution path there
config_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(os.path.dirname(config_directory),'_src'))

##################
## GET ARGUMENTS
##################
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of the base model configuration file that will be generated, with no extension .")
ap.add_argument("-p", "--path", required=True,
    help="path to the base model .h5 file")
ap.add_argument("-o", "--output-path", required=False,
    default=os.path.join(config_directory,'config_base_models'),
    help=" path to the directory where the configuration will be generated. Default : Configurations/config_base_models/ ")
ap.add_argument("-s", "--image-shape", required=False,
    default="[300,300,3]",
    help ="shape of the image that will be used for model training. Default : [300,300,3]")
ap.add_argument("-c", "--classes", required=False,
    default="[\'background\',\'cat\']",
    help ="Classes of the model. Default : ['background','cat']")
args = vars(ap.parse_args())

##################
## SETUP CONFIGURATION
##################
import gc
from utils import generateconfig as genc
## Generating data configuration
genc.setup_base_model_config(args['path'],args['name'],args['output_path'],
                        args['image_shape'],args['classes'])
