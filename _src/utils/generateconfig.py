import glob, sys, os
import shutil

def setup_base_model_config(model_path,config_file_name,output_path,
                        img_shape,
                        classes):

    configuration_txt = generate_model_config(model_path,
                                            img_shape,
                                            classes)
    new_file_config = open(os.path.join(output_path,config_file_name+'.py'),'w')
    new_file_config.write(configuration_txt)
    new_file_config.close()

    ##add the new config in the __init__ file of the config
    #TODO : only add it if it doesn't exist
    file_config = open(os.path.join(output_path,'__init__.py'),'a')
    file_config.write('from . import {} \n'.format(config_file_name))
    file_config.close()

def setup_model_config(config_file_name,
                        model_name,
                        model_output_path,
                        config_output_path,
                        img_shape,
                        classes):

    configuration_txt = generate_model_config(os.path.join(model_output_path,model_name),
                                            img_shape,
                                            classes)
    new_file_config = open(os.path.join(config_output_path,config_file_name+'.py'),'w')
    new_file_config.write(configuration_txt)
    new_file_config.close()

    ##add the new config in the __init__ file of the config
    #TODO : only add it if it doesn't exist
    file_config = open(os.path.join(config_output_path,'__init__.py'),'a')
    file_config.write('from . import {} \n'.format(config_file_name))
    file_config.close()

def setup_data_config(data_location,
                        data_config_name,
                        model_name,
                        output_path,
                        epochs,
                        batch_size,
                        test_batch_size
                        ):
    # # To be used only in the ssd/ level
    from utils import generateset as gens
    from utils import generateconfig as genc

    ### Generating the config
    conf=genc.generate_data_config(data_location,
                                    model_name,
                                    epochs,
                                    batch_size,
                                    test_batch_size)
    new_file_config = open(os.path.join(output_path,data_config_name+'.py'),'w')
    new_file_config.write(conf)
    new_file_config.close()

    ##add the new config in the __init__ file of the config
    #TODO : only add it if it doesn't exist
    file_config = open(os.path.join(output_path,'__init__.py'),'a')
    file_config.write('from . import {} \n'.format(data_config_name))
    file_config.close()

def generate_data_config(datapath,
                    modelName,
                    epochs,
                    batch_size,
                    test_batch_size):
    print('Generating new configuration ')
    config='import os \n \n'
    # config += 'ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))' + '\n'
    # config += 'DATA_DIR = \'{}\''.format(datapath.replace('\\','\\\\')) + '\n'
    config += 'DATA_DIR = \'{}\''.format(datapath.replace('\\','\\\\')) + '\n'
    config += 'IM_DIR = os.path.join(DATA_DIR,\'Images\')' +'\n'
    config += 'SETS_DIR = os.path.join(DATA_DIR,\'ImageSets\')' + '\n'
    config += 'LABELS_DIR = os.path.join(DATA_DIR,\'Annotations\')' + '\n'
    config += 'CHECKPOINT_NAME= \'{}'.format(modelName+'_checkpoint.h5\' \n')
    config += 'MODEL_NAME = \'{}'.format(modelName +'.h5\' \n')
    config += 'EPOCHS= {}'.format(epochs) + '\n'
    config += 'BATCH_SIZE={}'.format(batch_size) + '\n'
    config += 'TEST_BATCH_SIZE = {}'.format(test_batch_size) + '\n'
    return(config)

def generate_model_config(model_path,
                            img_shape,
                            classes):
    configuration_txt = ''
    configuration_txt+= 'PATH =\'{}'.format(model_path.replace('\\','\\\\')) +'.h5\''+ '\n'
    configuration_txt+= 'IMG_SHAPE = {}'.format(img_shape) + '\n'
    configuration_txt+= 'CLASSES = {}'.format(classes)
    return(configuration_txt)

def get_config_from_name(config_name):
    #To be used only in the ssd/ level
    import imp
    import importlib
    import config
    # config = importlib.import_module('..config')
    imp.reload(config)
    configuration= importlib.import_module('config.{}'.format(config_name))
    return configuration

def get_data_config_from_name(config_name):
        #To be used only in the ssd/ level
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'Configurations')
        working_dir = os.getcwd()
        sys.path.insert(0,config_path)
        import imp
        import importlib
        import config_models
        imp.reload(config_models)
        configuration= importlib.import_module('config_datas.{}'.format(config_name))
        sys.path.insert(0,working_dir)
        return configuration

def get_model_config_from_name(config_name):
    #To be used only in the ssd/ level
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'Configurations')
    working_dir = os.getcwd()
    sys.path.insert(0,config_path)
    import imp
    import importlib
    import config_models
    imp.reload(config_models)
    configuration= importlib.import_module('config_models.{}'.format(config_name))
    sys.path.insert(0,working_dir)
    return configuration

def get_base_model_config_from_name(config_name):
    #To be used only in the ssd/ level
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'Configurations')
    working_dir = os.getcwd()
    sys.path.insert(0,config_path)
    import imp
    import importlib
    import config_models
    imp.reload(config_models)
    configuration= importlib.import_module('config_base_models.{}'.format(config_name))
    sys.path.insert(0,working_dir)
    return configuration
