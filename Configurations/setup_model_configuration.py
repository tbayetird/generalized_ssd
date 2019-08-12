import sys
import os
#Previous implementations were made to be executed in the src folder so let's put our execution path there
config_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.join(os.path.dirname(config_directory),'_src')
sys.path.insert(0,src_directory)

##################
## GET ARGUMENTS
##################
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--configuration-name", required=True,
	help="name of the model configuration file that will be generated, with no extension .")
ap.add_argument("-m", "--model_name", required=False,
	default = None,
	help="name of the model .h5 file that will be generated, with no extension. Default = name given as configuration file name")
ap.add_argument("-omp", "--output-model-path", required=False,
	default = os.path.join(src_directory,'models'),
    help="path to the directory where the model will be generated after training or where it already is. Default : _src/models ")
ap.add_argument("-o", "--output-config-path", required=False,
    default=os.path.join(config_directory,'config_models'),
    help=" path to the directory where the configuration will be generated. Default : Configurations/config_models ")
ap.add_argument("-s", "--image-shape", required=False,
    default="[300,300,3]",
    help ="shape of the image that will be used for model training. Default : [300,300,3]")
ap.add_argument("-c", "--classes", required=False,
    default="[\'background\',\'cat\']",
    help ="Classes of the model. Default : ['background','cat']")
args = vars(ap.parse_args())

if args['model_name'] is None:
	model_name = args['configuration_name']
else:
	model_name = args['model_name']
##################
## SETUP CONFIGURATION
##################
import gc
from utils import generateconfig as genc
## Generating data configuration
genc.setup_model_config(args['configuration_name'],model_name,args['output_model_path'],
                        args['output_config_path'],args['image_shape'],args['classes'])
