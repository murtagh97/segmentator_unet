# Segmentator Unet

This repository contains a collection of scripts and notebooks of the project that examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the lung fields from a set of front view chest X-rays.

## Repository structure
├── README.md                       <- README for using this project.
├── arg_parser.py                   <- CLI parser with default arguments of the final model
├── data_generator.py               <- Loads the raw data and prepares tf.data.Datasets
├── data_visualizer.py              <- Visualizes the results and predictions
├── images                          <- Contains title image for the streamlit app
│   └── title_img.bmp
├── main.py                         <- Main script to train the custom model
├── model_object.py                 <- Final UNet model object, uses data_generator.py and data_visualizer.py
├── notebooks_colab                 <- Interactive Colab notebooks
│   ├── main_colab.ipynb            <- Interactive main notebook to experiment with the model
│   └── streamlit_app_colab.ipynb   <- Notebook to run the streamlit app via colab
├── requirements.txt                <- Requirements
├── streamlit_app.py                <- Interactive streamlit app of the project 
├── trained_model                   <- Final trained mdoel
│   ├── saved_model
│   │   ├── checkpoint
│   │   ├── unet_best_model
│   │   ├── unet_best_model.data-00000-of-00001
│   │   └── unet_best_model.index
│   ├── train
│   │   └── events.out.tfevents.1619037909.90dc556b6e20.5521.1445.v2
│   └── validation
│       └── events.out.tfevents.1619037922.90dc556b6e20.5521.4802.v2
└── utils                           <- Helper modules that are used during model training
    ├── __init__.py
    ├── loss_functions.py           <- Custom segmentation loss functions
    └── train_config.py             <- Metrics anc callbacks used during training
