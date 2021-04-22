import os
import random

import numpy as np
import streamlit as st 
import tensorflow as tf

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
    ('Main', 'Model Details', 'Data Augmentation', 'Training and Evaluation', 'Segmentation Results', 'Custom Segmentator')
    )
    
    if selected_box == 'Main':
        main_intro() 

    if selected_box == 'Model Details':
        model_info(unet_model, args)

    if selected_box == 'Data Augmentation':
        data_augmentation(unet_model)

    if selected_box == 'Training and Evaluation':
        training(unet_model)

    if selected_box == 'Segmentation Results':
        results(unet_model)

    if selected_box == 'Custom Segmentator':
        segmentator(unet_model, args) 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    dir_name = "trained_model"
    dir = os.path.join(dir_name, 'saved_model', 'unet_best_model',)

    unet_model = UnetModel(args)
    unet_model.prepare_data(args)
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
        * Try different data augmentation methods,
        * Plot the training procedure and evaluate the model on the respective datasets,
        * Inspect the predicted segmentations on the respective datasets,
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
    st.title("Data Augmentation")

    crop = st.sidebar.slider("Crop Size", 0.5, 1.0) 
    bright = st.sidebar.slider("Brightness Magnitude", 0.0, 1.0)
    rotation = st.sidebar.slider("Rotation Angle", 0.0, 2.0)

    n_exs = st.sidebar.number_input('Number of Examples', min_value= 0, max_value=None)

    if st.sidebar.button('Run Augmentation', key='button'): 

        for i in range(n_exs):

            image = model.plot_augmentation(
                crop = crop,
                bright = bright,
                angle = rotation,
                n_skip = i,
                size = (400, 980),
                st_mode = True
                )
            st.write(image, use_column_width=True)
    else:
        st.markdown("Press the button to generate augmentation.")

def training(model):

    st.title("Training and Evaluation")

    col1, col2 = st.beta_columns((5, 1))

    plot = st.sidebar.checkbox("Plot Training Process", value= True, key='check_0')
    eval = st.sidebar.checkbox("Evaluate Model", value= False, key='check_1')

    if plot:
        image = model.plot_learning_curves(
            size = (580,815),
            st_mode = True
            )
        col1.write(image, use_column_width=True)

    if eval:
        eval_selector = st.sidebar.radio(
            'Select Dataset', 
            ('Train Set', 'Dev Set', 'Test Set')
            )
        if st.sidebar.button('Run Evaluation', key='button'): 

            if eval_selector == 'Train Set':
                col2.markdown('***Evaluating on Train Set:***')
                loss, sdc, hdc, iou, acc = model.model.evaluate(model.wf_train, steps = model.steps_per_epoch, verbose=0)
                col2.markdown(f'Train Loss: {loss:.4f}  \n Train SDC: {sdc:.4f}  \n Train HDC: {hdc:.4f}  \n Train IoU: {iou:.4f}  \n Train Acc: {acc:.4f}')

            if eval_selector == 'Dev Set':
                col2.markdown('***Evaluating on Dev Set:***')
                loss, sdc, hdc, iou, acc = model.model.evaluate(model.wf_dev, steps = model.steps_per_epoch, verbose=0)
                col2.markdown(f'Dev Loss: {loss:.4f}  \n Dev SDC: {sdc:.4f}  \n Dev HDC: {hdc:.4f}  \n Dev IoU: {iou:.4f}  \n Dev Acc: {acc:.4f}')

            if eval_selector == 'Test Set':
                col2.markdown('***Evaluating on Test Set:***')
                loss, sdc, hdc, iou, acc = model.model.evaluate(model.wf_test, steps = model.steps_per_epoch, verbose=0)
                col2.markdown(f'Test Loss: {loss:.4f}  \n Test SDC: {sdc:.4f}  \n Test HDC: {hdc:.4f}  \n Test IoU: {iou:.4f}  \n Dev Acc: {acc:.4f}')
    else:
        col2.markdown("**Check the *Evaluate Model* box to start model evaluation.**")


def results(model):
    st.title("Segmentation Results")

    set_selector = st.sidebar.radio(
        'Select Dataset', 
        ('Train Set', 'Dev Set', 'Test Set')
        )

    if set_selector == 'Train Set':
        dataset = model.wf_train

    if set_selector == 'Dev Set':
        dataset = model.wf_dev

    if set_selector == 'Test Set':
        dataset = model.wf_test

    n_preds = st.sidebar.number_input('Number of Predictions', min_value= 0, max_value=None)

    if st.sidebar.button('Run Predictions', key='button'): 
        skip = random.randint(0,10)
        for i in range(n_preds):
            image = model.plot_predictions(
                dataset = dataset,
                n_skip = skip + i,
                size = (400, 980),
                st_mode = True
                )
            st.write(image, use_column_width=True)

    else:
        st.markdown("Press the button to generate predictions.")

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

        img_plotly = model.predict_single_img(
            img,
            size = (400,980),
            st_mode = True
            )
        
        st.write(
            img_plotly, 
            use_column_width=True
            )

if __name__ == "__main__":

    main(args)