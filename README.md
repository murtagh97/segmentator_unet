# Lung Fields Segmentator App 
This repository contains a project that examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the lung fields from a set of front view chest X-rays, with the final model deployed in the <a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live app</a> built with <a href="https://streamlit.io/" target="_blank">Streamlit</a>.

## Project summary

### Requirements
streamlit==0.80.0
plotly==4.14.3
matplotlib==3.4.1
ipython==7.22.0
Pillow==8.2.0
tensorflow==2.4.1
tensorflow_addons==0.12.1

### Dataset
Source of the data used for model training is <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">SCR dataset</a>. More information about the data and preprocessing can be found in the deployed app.

The data folder can be found **TODO**.

| Metric | Train | Dev | Test |
| --- | --- | --- | --- |
| Loss | 0. | 0. | 0. |
| Dice Coef. | 0. | 0. | 0. |
| IoU | 0. | 0. | 0. |

### Trained model
Contains the weights and training history of the final model.

### Model Scripts

| Script | Description |
| --- | --- |
| main.py | Runs the model training.|
| model_object.py | Contains the main *UnetModel* object that uses data_generator.py and inherits from data_visualizer.py.|
| data_generator.py | Contains the *DataGenerator* object that loads the raw data, runs the augmentation and prepares the train-dev-test tf.data pipelines|
| data_visualizer.py | Contains the *DataVisualizer* that plots the training procedure, segmentation results, etc.|
| arg_parser.py | Contains parser with the arguments that model training and interaction can be controlled with. The defaults are set to the parameters of the final model.|
| utils/loss_functions.py | Contains custom loss functions used for model training.|
| utils/train_config.py | Contains METRICS and CALLBACKS used for model training.|

### Colab Notebooks

| Notebook | Description |
| --- | --- |
| notebooks_colab/main_notebook.ipynb | Interactive notebook that allows to train a new model or load the trained model and display the training procedure and segmentation results.|
| notebooks_colab/streamlit_app_notebook.ipynb | Notebook that enables to run the local app streamlit_app_local.py via Colab.|

### Streamlit app 
| Versiion | Description |
| --- | --- |
| streamlit_app.py | Deployed live via |
| streamlit_app_local.py | Notebook that enables to run the local app streamlit_app_local.py via Colab.|
