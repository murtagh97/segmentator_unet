# Lung Fields Segmentation: Model and Deployment
<a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live app</a>

This repository contains a project that examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the lung fields from a set of front view chest X-rays given in the <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">SCR dataset</a>, with the final model deployed in the <a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live app</a> built with <a href="https://streamlit.io/" target="_blank">Streamlit</a>.

Feel free to reach out if you have any feedback or suggestions!

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
The SCR dataset can be found <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">here</a>. More information about the nature of the data and the preprocessing can be found in the deployed app.

Folders <a href="https://github.com/murtagh97/segmentator_unet/tree/main/images/test_samples" target="_blank">images/train_samples</a> and <a href="https://github.com/murtagh97/segmentator_unet/tree/main/images/train_samples" target="_blank">images/test_samples</a> contain X-ray samples from the SCR dataset train and test sets.

### Trained Model
Folder <a href="https://github.com/murtagh97/segmentator_unet/tree/main/trained_model" target="_blank">trained_model</a> contains the weights and the training history of the final model.
Achieved results are shown in the table below.

| Metric | Train | Dev | Test |
| --- | --- | --- | --- |
| Loss | 0.1089 | 0.1109 | 0.1166 |
| Dice Coef. | 0.9824 | 0.9807 | 0.9747 |
| IoU | 0.9650 | 0.9614 | 0.9505 |

Further information about the model architecture, training settings and hyperparameters can be found in the deployed app.

### Model Scripts

| Script | Description |
| --- | --- |
| main.py | Runs the model training.|
| model_object.py | Contains the main *UnetModel* object that uses data_generator.py and inherits from data_visualizer.py.|
| data_generator.py | Contains the *DataGenerator* object that loads the raw data, runs the augmentation and prepares the train-dev-test tf.data pipelines. |
| data_visualizer.py | Contains the *DataVisualizer* object that plots the training procedure, segmentation results, etc.|
| arg_parser.py | Contains parser with arguments that model training and inference can be controlled with. The defaults are set to the parameters of the final model.|
| utils/loss_functions.py | Contains custom loss functions used for the model training.|
| utils/train_config.py | Contains custom METRICS and CALLBACKS used for the model training.|

### Colab Notebooks

| Notebook | Description |
| --- | --- |
| notebooks_colab/main_notebook.ipynb | Interactive notebook that allows to train a new model from scratch, to load the trained model, to display the training procedure and the segmentation results.|
| notebooks_colab/streamlit_app_notebook.ipynb | Notebook that enables to run the local app streamlit_app_local.py on Colab.|

### Streamlit Deployments
| Version | Description |
| --- | --- |
| streamlit_app.py | Deployed <a href="https://share.streamlit.io/murtagh97/segmentator_unet/main" target="_blank">live</a>.|
| streamlit_app_local.py | Recommended and tested deployment on Colab via notebooks_colab/streamlit_app_notebook.ipynb. Works directly with the train-dev-test datasets that need to be created. Slower, but contains some additional features.|

## Acknowledgements
Many thanks to Streamlit Sharing for the free hosting! :upside_down_face:
