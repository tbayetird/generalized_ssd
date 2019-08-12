## Generalized single shot detector

This repository implements a new way to detect elements on image using a deep learning network called single shot detector (ssd), on windows, using keras.

First of all, great thanks to pierluigiferrari for his implementation in keras of the ssd method. We vastly use his work that can be found here : https://github.com/pierluigiferrari/ssd_keras

This project aims to make it easier to go through all the steps required to train your own object detector. You don't have to go into code really, all you have to do is to follow easy steps of configurations.

But anyone can dive into the code as it's pretty easy to understand and to follow all the workflow. More about it in the sections below.

### What can you do with this repository ?

In a few words :

- train your own object detection deep-learning model
- use your own trained ssd model to infere on data
- use a generic ssd model to infere on data

### Prerequisites

This project request from you that you have an operationnal working keras environment installed. In particular, working on anaconda on windows, the versions used for development and testing are as below :

python                    3.6.7
keras                     2.2.4                    
keras-applications        1.0.6                    
keras-preprocessing       1.0.5                    
keras-rl                  0.4.2                    
tensorflow-gpu            1.12.0

It is mandatory that you have a working keras environment to use this implementation.

### Installation

This repository doesn't work out of the box ; you'll have to get the ssd implementation from pierluigiferrari. Download the zip or pull his git and copy/paster all the folders in the src folder.

After this operation, your repository should look like this (folders only):
.
├── Configurations
│   ├── config_base_models
│   ├── config_datas
│   └── config_models
├── Dataset Formatting
│   └── labelimg
│       └── data
├── Inference
├── Training
└──\_src
    ├── bounding_box_utils
    ├── data_generator
    ├── eval_utils
    ├── example
    ├── inference_utils
    ├── keras_layers
    ├── keras_loss_function
    ├── misc_utils
    ├── models
    ├── outputs
    ├── ssd_encoder_decoder
    ├── test_utils
    ├── train_utils
    ├── unit_tests
    ├── utils
    │   └── labelimg
    │       └── data
    └── weights

There, you're ready. Be warry of the fact that some calls between files are hardcoded in the scripts. If you ever modifiy this architecture, you might break the calls and get errors. Do this only if you know what you're doing.

### How to understand and use this repository

All the source files (ssd implementation for pierluigiferrari, my own implementations on top of it, usefull scripts) are in the \_src directory. The models, weights, and outputs are per default in there too, but you will be able to change that.  

All the other folders are explicits and contains scripts that are named explicitly.

#### How to train a model

To train a model, you will initially need weights for the model you want to train. This is due to pierluigiferrari's implementation and will stay like this for now. Follow instructions on his github to downloads the weights you need for your model.

Once you've got the weights, you'll have to setup the configurations.

##### Configurations

Those are the core of the repository. The configurations may be the only files you might need to actually type by yourself, even though there are scripts to automatically generate them.

There are three types of configurations :
- The base models configurations : contains informations on what we call base models, i.e the weights downloaded before training. Only useful if you plan on training some models
- The model configurations : contains informations on models that will be trained or used for inference. Base model configurations can be used there if you want to use them for your detection.
- The data configurations : contains informations on the data for training : data paths, training parameters.

A valid configuration is a configuration that has everything right in it : right types, valids paths, ...
To train a model, you will need at least a valid base model configuration, a valid model configuration and a valid data configuration.

To use a model (infere on some data), you will only need a valid model configuration.

To see how you're supposed to setup the configurations, see full workflow below.

##### Launch training

To train a model, all you need is :
- a valid dataset (PASCALVOC architecure, all data labelled)
- valid configurations (data configuration, model configuration)

Then launch the train.py script in the TRAINING folder with the configurations names as parameters.

#### How to infere with a model

Easy, just have a valid model configuration and run the inference.py script with, as parameters :
- the path to your data
- the name of the model configuration you wish to use

### Full Workflow

In this section we'll explore all the workflow of object detection. We'll go through :
- data formatting
- configuration setup
- training
- testing
- inference

#### Data

Object detection only occurs on visual data ; here we'll only consider images and videos. As this implementation is pretty basic, for specifics needs you may need to get your hands dirty and dive into the code. But for every simple use cases, configurations will be your only worries here.


#### Data formatting

In order to train a model, we'll need images to train on. If all you have as data is videos, you'll need to extract images from this video to train your model.
Your images need to be organized in a proper way so that the configurations can automatically be generated, or you'll have to do everything by hand. In our case, this implementation only support PASCALVOC architecture. Do not worry, it's easy to set up. Let's go.

First, get all your images in a folder. For this example, we'll use the \_src/example folder :

example/
    ├── 000019.jpg
    ├── 000044.jpg
    ├── 000122.jpg
    ├── 000158.jpg
    ├── 000215.jpg
    ├── 000241.jpg
    ├── 000242.jpg
    ├── 000325.jpg
    ├── 000400.jpg
    ├── 000528.jpg
    ├── 000605.jpg
    ├── 000619.jpg
    ├── 000876.jpg
    ├── 000908.jpg
    ├── 000912.jpg
    ├── 001001.jpg
    ├── 001015.jpg
    ├── 001072.jpg
    ├── 001156.jpg
    ├── 001233.jpg
    ├── 001260.jpg
    ├── 001553.jpg
    ├── 001607.jpg
    ├── 001669.jpg
    ├── 001732.jpg
    ├── 001747.jpg
    ├── 001888.jpg
    ├── 001927.jpg
    ├── 001982.jpg
    └── 002067.jpg

Those are images extracted from the PascalVOC 2007 devkit. They are not yet under the PascalVOC architecture though. This architecture need :
- an Images folder, containing all images
- an ImageSets folder, containing the sets (train.txt, validation.txt, test.txt, three text files containing the IDs of the image going into training, validation or test dataset)
- an Annotations folder, containing a label (.xml) file for every image in the Images folder.

You can either create those by yourself (be warry, error here will raise errors while running later) or use the from_image_folder-to_PAscalVOC_architecure.py script in the Dataset Formating folder.

This script takes as argument :
 - Path (-p) : the path to the image folder that will be formatted into a PAscalVOC-like architecture

 In our case, the script will be launched like that :

 python Path_to_generalized_ssd\Dataset-Formatting\from_image_folder-to_PAscalVOC_architecure.py -p Path_to_generalized_ssd\_src\example

There, your data is formatted, your sets are generated. You now need to label your data.

To label your data, use the label_img executable in the labelimg folder. This application made by tzutalin (https://github.com/tzutalin/labelImg) allows you to easilly label your data.

Launch the application ; open your images folder (in our case, \_src/example/Images) ; change the save directory to where you want to export your labels (in our case : \_src/example/Annotations)

For each image, create a new rect for each object you want to detect. In our case, we're gonna build a cat detector, so we're labelling our images as follow :

[INSERT IMAGE HERE]

Be sure to verify every image, even those without any annotation, or when training you'll have errors (press space bar to verify image after you're done annotating with rectangles)

Once you're done, you can close the application and go to the next step.

#### Setting up Configuration

The configurations are the most important part here. Everything, from training through inference, is done thanks to the configurations. Be warry to set them up correctly.

The base model configuration is as it follows :
PATH = 'Path_to_generalized_ssd\\\_src\\models\\base_model_example.h5'
IMG_SHAPE = [300,300,3]
CLASSES = ['background','cat']

The PATH leads to the base model weights, i.e the ones you've downloaded (or just use the examples ones)
The IMG_SHAPE is the shape of the images that are used for training. It depends on the network, and those images are of size 300x300 with colors (r,g,b) in our example.
The CLASSES corresponds to the initial CLASSES for the base_model. But it's not really usefull here.


The model configuration is as it follows :
PATH = 'Path_to_generalized_ssd\\\_src\\models\\model_example.h5'
IMG_SHAPE = [300,300,3]
CLASSES = ['background','cat']

The configuration is roughly the same as the base model, except you should put here the classes you want to train your model to detect. Beware, always include background as a classe.


The data configuration is as follows :
DATA_DIR = 'Path_to_generalized_ssd\\\_src\\example\\'
IM_DIR = os.path.join(DATA_DIR,'Images')
SETS_DIR = os.path.join(DATA_DIR,'ImageSets')
LABELS_DIR = os.path.join(DATA_DIR,'Annotations')
CHECKPOINT_NAME= 'example_model_checkpoint.h5'
MODEL_NAME = 'example_model.h5'
EPOCHS= 40
BATCH_SIZE=16
TEST_BATCH_SIZE = 1

DATA_DIR corresponds to the folder containing your data.
IM_DIR is the folder containing the images
SETS_DIR is the folder containing the sets
LABELS_DIR is the folder containing the labels.
CHECKPOINT_NAME is the name the intermediary model should be named with. The intermediary model is the state of the weights after each epoch.
MODEL_NAME is the name of the model that will be saved after training
EPOCHS : number of epochs
BATCH_SIZE : size of the batch for training
TEST_BATCH_SIZE : size of batch size when testing.

Once your configurations are set up, you can go through training and using your models.  

#### Training

Once your configurations are all valid, you can start training your models. This is easy : just run the train.py script in the Training folder.

This script takes as argument :
 - data-configuration (-d) : 	name of the configuration file related to the dataset to be trained.
 - model-configuration (-m) : name of the configuration file related to the model to be trained
 - base-model-configuration (-b) : name of the configuration file related to the base model you want to use

 In our case, the script will be launched like that :

 python Path_to_generalized_ssd\Training\train.py -d example_data -m example_model -b example_base_model

 The training will then launch and run until termination. The weights of the trained model will be saved in the \_src/models folder.

#### Testing

[TO COME]

#### Infering on your data

To infere on your data, all you need is a valid model configuration.
Run the script you want in the Inference folder.

These scripts share commons arguments :
- model-configuration (-m) : name of the model configuration to use for inference
- path (-p) : path to the data
- confidence-threshold (-c) : the confidence threshold used when sorting predictions after inference.
