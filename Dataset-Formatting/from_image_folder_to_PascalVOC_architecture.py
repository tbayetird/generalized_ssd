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
ap.add_argument("-p", "--path", required=True,
    help="path to the image folder that will be formatted into a PascalVOC-like architecture")
args = vars(ap.parse_args())

###################
## FORMAT DIRECTORY
###################
from utils import generateset as gens
gens.PascalVOC_generate(args['path'])
