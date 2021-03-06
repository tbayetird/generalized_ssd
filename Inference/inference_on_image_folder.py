import sys
import os
#Previous implementations were made to be executed in the src folder so let's put our execution path there
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'_src'))

##################
## GET ARGUMENTS
##################
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-configuration", required=True,
	help="name of the configuration file of the model we want to infere on")
ap.add_argument("-p", "--path", required=True,
	help="path to the image folder we want to infere on")
ap.add_argument("-c", "--confidence-threshold", required=False,
    default=0.5,
    help=" confidence threshold to be used during the inference")
ap.add_argument("-d", "--display", required=False,
    default=True,
    help=" Boolean ; True if you want to display all the results, False else. initialized with True")

args = vars(ap.parse_args())

from inference_utils.ssd_inference import inference_on_folder
from utils.generateconfig import get_model_config_from_name

model_configuration=get_model_config_from_name(args['model_configuration'])
inference_on_folder(
                    model_configuration,
                    args['path'],
                    args['display'],
                    args['confidence_threshold']
)
