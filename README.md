# segmentator_unet 
This repository contains a project that examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the lung fields from a set of front view chest X-rays, with the final model deployed in the <a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live app</a> built with <a href="https://streamlit.io/" target="_blank">Streamlit</a>.

## Project summary
### Dataset
Source of the data used for model training: <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">SCR dataset</a>. Data information and preprocessing is described in the deployed app. 

### Model Scripts

| Script | Description |
| --- | --- |
| main.py | Runs model training|
| model_object.py | Contains the main *UnetModel* object that uses data_generator.py and inherits data_visualizer.py |
| data_generator.py | Contains the *DataGenerator* object that loads the raw data, runs the augmentation and prepares the train-dev-test tf.data pipelines|
| data_visualizer.py | Contains the *DataVisualizer* that plots the training procedure, segmentation results, etc.|
| arg_parser.py | Contains parser with the arguments that model training and interaction can be controlled. Defaults are set to the parameters of the final model.|

## Streamlit app 
TODO
