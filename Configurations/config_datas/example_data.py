import os 
 
DATA_DIR = 'D:\\workspace\\object-detection\\generalized-single-shot-detector\\_src\\example\\'
IM_DIR = os.path.join(DATA_DIR,'Images')
SETS_DIR = os.path.join(DATA_DIR,'ImageSets')
LABELS_DIR = os.path.join(DATA_DIR,'Annotations')
CHECKPOINT_NAME= 'example_model_checkpoint.h5' 
MODEL_NAME = 'example_model.h5' 
EPOCHS= 40
BATCH_SIZE=16
TEST_BATCH_SIZE = 1
