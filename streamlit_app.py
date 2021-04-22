import os
import random

import numpy as np
import streamlit as st 
import tensorflow as tf
import tensorflow_addons as tfa
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
    'Select Section',
    ('Main', 'Model Details', 'Training and Results', 'Upload Augmentation', 'Upload Segmentation')
    )
    
    if selected_box == 'Main':
        main_intro() 

    if selected_box == 'Model Details':
        model_info(unet_model, args)

    if selected_box == 'Training and Results':
        training(unet_model)

    if selected_box == 'Data Augmentator':
        data_augmentation(unet_model)

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
        This app examines the use of <a href="https://arxiv.org/abs/1505.04597" target="_blank">UNet</a> model to segment the 
        lung fields from a set of front view chest X-rays given in the <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">SCR dataset</a>.
        """,
        unsafe_allow_html=True
        )
    
    col1, col2 = st.beta_columns((2, 1))
    col1.markdown(
        '''
        Individual app sections allow user to: 
        * See the details of the final model and the underlying dataset,
        * Display the training procedure and the model results on the respective datasets,
        * Upload an image and try different data augmentation methods,
        * Upload an image and predict the resulting segmentation.
        '''
        )
    col1.markdown(
        '''
        Feel free to reach out if you have any feedback or suggestions!
        '''
        )
    
    title_img = Image.open('images/title_img.bmp')
    col2.image(title_img)

def model_info(model, args):
    st.title("Model Details")

    exp_augment = st.beta_expander("Dataset Information")
    exp_augment.markdown(
        f'* *Dataset source:* <a href="https://www.isi.uu.nl/Research/Databases/SCR/" target="_blank">SCR dataset</a>,  \n'
        f'* *Train set:*  124 lung fields chest X-rays and their ground-truth masks,  \n'
        f'* *Test set:*  123 lung fields chest X-rays and their ground-truth masks,   \n'
        f'* *Train-dev split:*  0.85.  \n',
        unsafe_allow_html=True
        )

    exp_architect = st.beta_expander("Model Architecture")
    exp_architect.markdown(
        f'* *Architecture:* <a href="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" target="_blank">UNet</a>,  \n'
        f'* *Upsample method:* transposed convolutions,  \n'
        f'* *Number of trainable parameters:* {model.trainable_params},  \n'
        f'* *Max filter size:* {args.max_filter_size}.  \n',
        unsafe_allow_html=True
        )
    
    exp_augment = st.beta_expander("Preprocessing and Data Augmentation")
    exp_augment.markdown(
        f'* *Preprocessing:*  \n'
        f'  * images, masks resized into {args.mask_size}x{args.mask_size},  \n'
        f'  * images, masks normalized into [0,1],  \n'
        f'  * masks binarized,  \n'
        f'* *Random central crop, crop size:* {args.crop},  \n'
        f'* *Random brightness adjustment, brightness rate:* {args.bright}.  \n'
        )
    
    exp_train = st.beta_expander("Training Settings")
    exp_train.markdown(
        f'* *Batch size:* {args.batch_size},  \n'
        f'* *Number of epochs:* {args.n_epochs},  \n'
        f'* *Optimiser:* Adam,  \n'
        f'* *Base learning rate:* {args.base_lr},  \n'
        f'* *Learning rate schedule:* reduce learning rate on plateau,  \n'
        f'* *Loss function:* Dice loss,  \n'
        f'* *Monitored metrics:* Dice coefficient, intersection over union, accuracy.  \n'
        )
    
    exp_hyper = st.beta_expander("Hyperparameters")
    exp_hyper.markdown(
        f'* *Dropout:* {args.dropout},  \n'
        f'* *L2 regularization:* {args.l2}.  \n'
        )
    
def training(model):

    st.title("Training and Results")

    col1, col2 = st.beta_columns((5, 1))

    plot = st.sidebar.checkbox("Display Training Procedure", value= True, key='check_0')
    eval = st.sidebar.checkbox("Display Model Results", value= False, key='check_1')

    if plot:
        image = model.plot_learning_curves(
            size = (580,815),
            st_mode = True
            )
        col1.write(image, use_column_width=True)
    else:
        col1.markdown("**Check the *Display Training Procedure* box to display the training procedure.**")

    if eval:
        eval_selector = st.sidebar.radio(
            'Select Dataset', 
            ('Train Set', 'Dev Set', 'Test Set')
            )
        if st.sidebar.button('Display Results', key='button'): 

            if eval_selector == 'Train Set':
                col2.markdown('***Evaluation on Train Set:***')
                col2.markdown(f'Train Loss: 0.1089  \n Train SDC: 0.9824  \n Train HDC: 0.9830  \n Train IoU: 0.9650  \n Train Acc: 0.3817')

            if eval_selector == 'Dev Set':
                col2.markdown('***Evaluation on Dev Set:***')
                col2.markdown(f'Dev Loss: 0.1109  \n Dev SDC: 0.9807  \n Dev HDC: 0.9809  \n Dev IoU: 0.9614  \n Dev Acc: 0.2982')

            if eval_selector == 'Test Set':
                col2.markdown('***Evaluation on Test Set:***')
                col2.markdown(f'Test Loss: 0.1166  \n Test SDC: 0.9747  \n Test HDC: 0.9749  \n Test IoU: 0.9505  \n Dev Acc: 0.2938')
    else:
        col2.markdown("**Check the *Display Model Results* box to display the model results on the selected dataset.**")
    
def data_augmentation(model):

    st.title("Data Augmentator")

    st.markdown("Upload a front view chest X-ray image of lung fields to display different data augmentation methods.")

    uploaded_img = st.file_uploader(
        "Upload Image", 
        type = ['bmp', 'jpg', 'jpeg', 'png'], 
        accept_multiple_files= False
        )
    
    st.markdown(
        """
         Additional examples of chest X-ray images can be found, e.g., 
         <a href="https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg" target="_blank">here</a>,
         <a href="https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png" target="_blank">here</a> and 
         <a href="https://www.kaggle.com/tolgadincer/labeled-chest-xray-images" target="_blank">here</a> .
        """,
        unsafe_allow_html=True
        )
    
    if uploaded_img is not None:
        crop = st.sidebar.slider("Crop Size", 0.5, 1.0) 
        bright = st.sidebar.slider("Brightness Magnitude", 0.0, 1.0)
        rotation = st.sidebar.slider("Rotation Angle", 0.0, 2.0)
        if st.sidebar.button('Display Augmentation', key='button'):

            img = Image.open(uploaded_img)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.image.resize(img, [args.mask_size, args.mask_size])
            img = ( tf.cast(img, tf.float32) / 255.0)
            img = np.expand_dims(img[:,:,0], axis = 2)

            img_cropped = tf.image.central_crop(img, central_fraction = crop)
            img_resized = tf.image.resize_with_pad(img_cropped, 256, 256)

            img_bright = tf.image.adjust_brightness(img, delta = bright)
            img_bright = tf.clip_by_value(img_bright, 0.0, 1.0)

            img_flipped = tf.image.flip_left_right(img)

            img_rot = tfa.image.rotate(img, rotation*np.pi, interpolation = 'nearest', fill_mode = 'reflect')

            display_list = [img, img_flipped, img_resized, img_bright, img_rot]
            title_list = [f'Input Image', 'Vertical Flip', f'Crop: {crop}', f'Bright: {bright}', f'Rotation: {rotation:.2f}\u03C0']


            plt.figure(figsize = (15,15))
            for i in range(len(display_list)):
                plt.subplot(1, len(display_list), i+1)
                plt.title(title_list[i])
                plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap = "gray")
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
            
            st.pyplot(plt)

def segmentator(model, args):
    st.title("Upload Segmentation")
    
    st.markdown("Upload a front view chest X-ray image of lung fields to display the predicted segmentation result.")

    uploaded_img = st.file_uploader(
        "Upload Image", 
        type = ['bmp', 'jpg', 'jpeg', 'png'], 
        accept_multiple_files= False
        )

    st.markdown(
        """
        Examples of chest X-ray images from the SCR train and test sets can be found 
        <a href="https://github.com/murtagh97/segmentator_unet/tree/main/images" target="_blank">here</a> .  \n
        
        Additional out-of-sample examples of chest X-ray images can be found, e.g., 
        <a href="https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg" target="_blank">here</a>,
        <a href="https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png" target="_blank">here</a> and 
        <a href="https://www.kaggle.com/tolgadincer/labeled-chest-xray-images" target="_blank">here</a> .
        """,
        unsafe_allow_html=True
        )

    if uploaded_img is not None:

        img = Image.open(uploaded_img)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, [args.mask_size, args.mask_size])
        img = ( tf.cast(img, tf.float32) / 255.0)
        img = np.expand_dims(img[:,:,0], axis = 2)

        one_img_batch = img[tf.newaxis, ...]

        pred_mask = model.model.predict(one_img_batch)
        pred_mask = np.reshape(pred_mask, (args.mask_size, args.mask_size, 1))
        pred_mask = np.array(pred_mask > 0.5, dtype=int)

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
        
        st.pyplot(plt)

if __name__ == "__main__":

    main(args)