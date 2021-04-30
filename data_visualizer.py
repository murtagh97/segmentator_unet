import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import plotly.express as px
import plotly.graph_objects as go

from arg_parser import args
from utils import loss_functions
from IPython.display import display
from plotly.subplots import make_subplots

class DataVisualiser:

    def __init__():
            pass
    
    def plot_learning_curves(
        self,
        size = (800, 1400),
        st_mode = False
        ):
        """Plot the training procedure curves.
        """
        acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        dsc = self.history['soft_dice_coef']
        val_dsc = self.history['val_soft_dice_coef']
        iou = self.history['iou']
        val_iou = self.history['val_iou']

        fig = make_subplots(
            rows=2, 
            cols=2, 
            subplot_titles=("Training and Development Loss", "Training and Development Soft Dice Coef.", "Training and Development IoU", "Training and Development Accuracy")
            )

        fig.add_trace(go.Scatter(y=loss, mode="lines", name = "Train Loss", line_color = "#ffa4a4"), row=1, col=1)
        fig.add_trace(go.Scatter(y=val_loss, mode="lines", name = "Dev Loss", line_color="#88d8b0"), row=1, col=1) 

        fig.add_trace(go.Scatter(y=dsc, mode="lines", name = "Train SDC", line_color = "#ffa4a4"), row=1, col=2)
        fig.add_trace(go.Scatter(y=val_dsc, mode="lines", name = "Dev SDC", line_color="#88d8b0"), row=1, col=2)

        fig.add_trace(go.Scatter(y=iou, mode="lines", name = "Train IoU", line_color = "#ffa4a4"), row= 2, col = 1)
        fig.add_trace(go.Scatter(y=val_iou, mode="lines", name = "Dev IoU", line_color="#88d8b0"), row = 2, col = 1) 

        fig.add_trace(go.Scatter(y=acc, mode="lines", name = "Train Acc", line_color = "#ffa4a4"), row=2, col=2)
        fig.add_trace(go.Scatter(y=val_acc, mode="lines", name = "Dev Acc", line_color="#88d8b0"), row=2, col=2)

        fig.update_layout(
            title= "Training Process",
            title_x= 0.5,
            template = "plotly_white",
            height = size[0],
            width = size[1],
            legend=dict(
                y=0.5,
                x=1
            )
        )

        fig['layout']['xaxis']['title']='Epochs'
        fig['layout']['xaxis2']['title']='Epochs'
        fig['layout']['xaxis3']['title']='Epochs'
        fig['layout']['xaxis4']['title']='Epochs'

        fig['layout']['yaxis']['title']='Loss'
        fig['layout']['yaxis2']['title']='SDC'
        fig['layout']['yaxis3']['title']='IoU'
        fig['layout']['yaxis4']['title']='Acc'

        if st_mode:
            return fig
        else:
            fig.show()

    @staticmethod
    def _get_bin_mask(
        output, 
        mask_size = args.mask_size
        ):
        """Function to display predicted mask.
        """
        data = np.reshape(output, (mask_size, mask_size, 1))
        mask = np.array(data > 0.5, dtype=int)
        return mask
    
    def _display_plots(
        self,
        display_list,
        title_list,
        size
        ):
        """Display the images from the display_list.
        """
        n_plots = len(display_list)
        fig = make_subplots(
            rows=1, cols= n_plots, 
            shared_yaxes=True,
            subplot_titles= title_list,
            horizontal_spacing = 0.05,
            vertical_spacing = 0.05
        )

        for i in range(n_plots):
            fig.add_trace(px.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i])).data[0], row=1, col= i+1)

        layout = px.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]), color_continuous_scale='gray').layout
        fig.layout.coloraxis = layout.coloraxis
        fig.update_xaxes(**layout.xaxis.to_plotly_json(), showticklabels=False)
        fig.update_yaxes(**layout.yaxis.to_plotly_json(), showticklabels=False)
        fig.update_layout(
            coloraxis_showscale=False,
            height = size[0],
            width = size[1]
            )

        return fig

    def plot_predictions(
        self,  
        dataset,
        n_take = 1, 
        n_skip = 0,
        size = (500,1000),
        st_mode = False
        ):
        """Display the final predictions: input image, input mask, predicted mask, highlighted masks difference.
        """
        if dataset == self.wf_train:
            name = "Train"
        elif dataset == self.wf_dev:
            name = "Dev"
        else:
            name = "Test"

        for i in range(n_take):
            img, mask = next(
                iter(
                    dataset.unbatch().skip(n_skip + i)
                    )
                )

            one_img_batch = img[tf.newaxis, ...]
            pred_mask = self.model.predict(one_img_batch)

            dicecoef = loss_functions.soft_dice_coef(mask, pred_mask)

            mask = self._get_bin_mask(mask)
            pred_mask = self._get_bin_mask(pred_mask)
            display_list = [img, mask, pred_mask, pred_mask - mask]
            title = [name + ' Input Image', 'True Mask', 'Predicted Mask with SDC: {:.4f}'.format(dicecoef), 'Highlighted Difference']

            fig = self._display_plots(
                display_list,
                title,
                size = size
                )

            if st_mode:
                return fig
            else:
                fig.show()

    def predict_single_img(
        self,
        img,
        size = (500,1000),
        st_mode = False
        ):
        """Display the final prediction on one image: input image, predicted mask, segmented result.
        """

        one_img_batch = img[tf.newaxis, ...]
        pred_mask = self.model.predict(one_img_batch)
        pred_mask = self._get_bin_mask(pred_mask)

        fig = self._display_plots(
            [img, pred_mask, img*pred_mask], 
            ["Uploaded Image", "Predicted Mask", "Segmented Result"],
            size = size    
            )
        if st_mode:
            return fig
        else:
            fig.show()

    def plot_augmentation(
        self, 
        crop = 0.7,
        bright = 0.25,
        angle = 0.25,
        n_take = 1, 
        n_skip = 0,
        size = (500, 1000),
        st_mode = False
        ):
        """Function to illustrate the effects of data augmentation.
        """
        
        for i in range(n_take):
            img, _ = next(
                iter(
                    self.wf_test.unbatch().skip(n_skip + i)
                    )
                )
            
            img_cropped = tf.image.central_crop(img, central_fraction = crop)
            img_resized = tf.image.resize_with_pad(img_cropped, 256, 256)

            img_bright = tf.image.adjust_brightness(img, delta = bright)
            img_bright = tf.clip_by_value(img_bright, 0.0, 1.0)

            img_flipped = tf.image.flip_left_right(img)

            img_rot = tfa.image.rotate(img, angle*np.pi, interpolation = 'nearest', fill_mode = 'reflect')

            display_list = [img, img_flipped, img_resized, img_bright, img_rot]
            title = [f'Input Image', 'Vertical Flip', f'Crop: {crop}', f'Bright: {bright}', f'Rotation: {angle:.2f}\u03C0']

            fig = self._display_plots(
                display_list,
                title,
                size = size
                )

            if st_mode:
                return fig
            else:
                fig.show()

    def plot_model(
        self
        ):
        """Display the tf.keras model architecture.
        """
        model_img = tf.keras.utils.plot_model(self.model, to_file= 'unet_model_architecture.png', show_shapes=True)
        display(model_img)
