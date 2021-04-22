# Lung Fields Segmentator: Model and Deployment
<a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live app</a>

This repository contains a project that examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the lung fields from a set of front view chest X-rays, with the final model deployed in the <a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live app</a> built with <a href="https://streamlit.io/" target="_blank">Streamlit</a>.

## Project Summary
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Trained Model](#trained-model)
* [Model Scripts](#model-scripts)
* [Colab Notebooks](#colab-notebooks)
* [Streamlit Deployments](#streamlit-deployments)

### Requirements
Please check the <a href="https://github.com/murtagh97/segmentator_unet/blob/main/requirements.txt" target="_blank">requirements.txt</a> file for the current version.

### Dataset
Source of the data used for model training is <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">SCR dataset</a>. More information about the data and preprocessing can be found in the deployed app.

The data folder can be found **TODO**.

### Trained Model
Folder *trained_model* contains the weights and training history of the final model.

Results
| Metric | Train | Dev | Test |
| --- | --- | --- | --- |
| Loss | 0.1089 | 0.1109 | 0.1166 |
| Dice Coef. | 0.9824 | 0.9807 | 0.9747 |
| IoU | 0.9650 | 0.9614 | 0.9505 |

Information about the model architecture, training settings and hyperparameters can be found in the deployed app.

### Model Scripts

| Script | Description |
| --- | --- |
| main.py | Runs the model training.|
| model_object.py | Contains the main *UnetModel* object that uses data_generator.py and inherits from data_visualizer.py.|
| data_generator.py | Contains the *DataGenerator* object that loads the raw data, runs the augmentation and prepares the train-dev-test tf.data pipelines|
| data_visualizer.py | Contains the *DataVisualizer* that plots the training procedure, segmentation results, etc.|
| arg_parser.py | Contains parser with the arguments that model training and inference can be controlled with. The defaults are set to the parameters of the final model.|
| utils/loss_functions.py | Contains custom loss functions used for model training.|
| utils/train_config.py | Contains METRICS and CALLBACKS used for model training.|

### Colab Notebooks

| Notebook | Description |
| --- | --- |
| notebooks_colab/main_notebook.ipynb | Interactive notebook that allows to train a new model or to load the trained model, display the training procedure and the segmentation results.|
| notebooks_colab/streamlit_app_notebook.ipynb | Notebook that enables to run the local app streamlit_app_local.py on Colab.|

### Streamlit Deployments
| Version | Description |
| --- | --- |
| streamlit_app.py | Deployed <a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live</a>.|
| streamlit_app_local.py | Recommended and tested deployment on Colab via notebooks_colab/streamlit_app_notebook.ipynb. Works directly with the train-dev-test datasets that need to be created. Slower, but contains some additional features.|

## Acknowledgements
Many thanks to Streamlit Sharing for free hosting! :upside_down_face:
