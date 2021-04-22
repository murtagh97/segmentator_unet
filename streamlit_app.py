import os
import random

import numpy as np
import streamlit as st 
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from arg_parser import args
from model_object import UnetModel

def main(args):

    st.set_page_config(
        page_title='Lungs Segmentation', 
        layout='wide', 
        initial_sidebar_state='expanded'
        )
    
    unet_model = load_model()

    selected_box = st.sidebar.selectbox(
    'Select Component',
    ('Main', 'Model Details', 'Data Augmentation', 'Training Process', 'Custom Segmentator')
    )
    
    if selected_box == 'Main':
        main_intro() 

    if selected_box == 'Model Details':
        model_info(unet_model, args)

    if selected_box == 'Data Augmentation':
        data_augmentation(unet_model)

    if selected_box == 'Training Process':
        training(unet_model)

    if selected_box == 'Custom Segmentator':
        segmentator(unet_model, args) 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    dir_name = "trained_model"
    dir = os.path.join(dir_name, 'saved_model', 'unet_best_model',)

    unet_model = UnetModel(args)
    unet_model.load_best_model(args, dir)
    unet_model.get_trainable_params()

    return unet_model

def main_intro():
    st.title("Lung Fields Segmentation from Chest X-rays")
    st.markdown(
        """
        by Július Rábek  \n

        <a href="https://github.com/murtagh97/segmentator_unet" target="_blank">GitHub</a> <a href="https://www.linkedin.com/in/julius-rabek/" target="_blank">LinkedIn</a>
        """,
        unsafe_allow_html=True
        )


    st.markdown(
        """
        This app examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the lung fields from a set of front view chest X-rays.
        """,
        unsafe_allow_html=True
        )
    
    col1, col2 = st.beta_columns((2, 1))
    col1.markdown(
        '''
        It allows user to: 
        * See the details of the final model,
        * Plot the training procedure,
        * Try different data augmentation methods on the uploaded image,
        * Upload an image and predict the custom segmentation.
        '''
        )
    
    title_img = Image.open('images/title_img.bmp')
    col2.image(title_img)

def model_info(model, args):
    st.title("Model Details")

    exp_architect = st.beta_expander("Model Architecture")
    exp_architect.markdown(
        f'* <a href="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" target="_blank">UNet</a>,  \n'
        f'* Number of trainable parameters: {model.trainable_params},  \n'
        f'* Max filter size: {args.max_filter_size}.',
        unsafe_allow_html=True
        )
    
    exp_augment = st.beta_expander("Data Augmentation")
    exp_augment.markdown(
        f'* Random central crop, crop size: {args.crop},  \n'
        f'* Random brightness adjustment, brightness rate: {args.bright}.  \n'
        )
    
    exp_train = st.beta_expander("Training Settings")
    exp_train.markdown(
        f'* Batch size: {args.batch_size},  \n'
        f'* Number of epochs: {args.n_epochs},  \n'
        f'* Optimiser: Adam,  \n'
        f'* Base learning rate: {args.base_lr},  \n'
        f'* Learning rate schedule: reduce learning rate on plateau,  \n'
        f'* Loss function: Dice loss,  \n'
        f'* Monitored metrics: Dice coefficient, intersection over union, accuracy.  \n'
        )
    
    exp_hyper = st.beta_expander("Hyperparameters")
    exp_hyper.markdown(
        f'* Dropout: {args.dropout},  \n'
        f'* L2 regularization: {args.l2}.  \n'
        )
    
def data_augmentation(model):
    pass
    # st.title("Data Augmentation")

    # crop = st.sidebar.slider("Crop Size", 0.5, 1.0) 
    # bright = st.sidebar.slider("Brightness Magnitude", 0.0, 1.0)
    # rotation = st.sidebar.slider("Rotation Angle", 0.0, 2.0)

    # n_exs = st.sidebar.number_input('Number of Examples', min_value= 0, max_value=None)

    # if st.sidebar.button('Run Augmentation', key='button'): 

    #     for i in range(n_exs):

    #         image = model.plot_augmentation(
    #             crop = crop,
    #             bright = bright,
    #             angle = rotation,
    #             n_skip = i,
    #             size = (400, 980),
    #             st_mode = True
    #             )
    #         st.write(image, use_column_width=True)
    # else:
    #     st.markdown("Press the button to generate augmentation.")

def training(model):

    st.title("Training Process")

    plot = st.sidebar.checkbox("Plot Training Process", value= True, key='check_0')

    if plot:
        image = model.plot_learning_curves(
            size = (580,1000),
            st_mode = True
            )
        st.write(image, use_column_width=True)
    else:
        st.markdown("**Check the *Plot Model* box to show the training process.**")

def segmentator(model, args):
    st.title("Custom Segmentator")
    
    st.markdown("Upload a front view chest X-ray image of lung fields to generate custom segmentation.")

    st.markdown(
        """
         Samples of chest X-ray images can be found, e.g., 
         <a href="https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg" target="_blank">here</a>,
         <a href="https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png" target="_blank">here</a> and 
         <a href="https://www.kaggle.com/tolgadincer/labeled-chest-xray-images" target="_blank">here</a> .
        """,
        unsafe_allow_html=True
        )

    uploaded_img = st.file_uploader(
        "Upload Image", 
        type = ['bmp', 'jpg', 'jpeg', 'png'], 
        accept_multiple_files= False
        )

    if uploaded_img is not None:

        img = Image.open(uploaded_img)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, [args.mask_size, args.mask_size])
        img = ( tf.cast(img, tf.float32) / 255.0)
        img = np.expand_dims(img[:,:,0], axis = 2)

        one_img_batch = img[tf.newaxis, ...]
        pred_mask = model.model.predict(one_img_batch)

        display_list = [img, pred_mask, img*pred_mask]
        title_list = ["Input Image", "Predicted Mask", "Segmented Result"]

        plt.figure(figsize = (15,15))
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title_list[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap = "gray")
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        
        st.write(
            plt, 
            use_column_width=True
            )

if __name__ == "__main__":

    main(args)