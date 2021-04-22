import numpy as np
import tensorflow as tf

import os
import pickle

from utils import loss_functions
from utils.train_config import METRICS, CALLBACKS
from data_generator import DataGenerator
from data_visualizer import DataVisualiser

class UnetModel(DataVisualiser):
    
    def __init__(
        self,args
        ):
        
        self.model = None

        self.wf_train = None
        self.wf_dev = None
        self.wf_test = None

        self.history = None

    def prepare_data(
        self, args
        ):

        self.wf_train, self.wf_dev, self.wf_test = DataGenerator.get_tf_wf(
            train_dev_split = args.train_dev_split,
            shuffle_size = args.shuffle,
            epochs = args.n_epochs,
            batch_size = args.batch_size,
            mask_size = args.mask_size,
            flip = args.flip,
            crop = args.crop,
            bright = args.bright,
            rot_angle = args.rot_angle,
            random_seed = args.random_seed
        )

        self.steps_per_epoch = DataGenerator.ds_size_train(args.train_dev_split) // args.batch_size

    def print_ds_info(
        self,args
        ):
        print(f'Train set size: {DataGenerator.ds_size_train(args.train_dev_split)}')
        print(f'Dev set size: {DataGenerator.ds_size_dev(args.train_dev_split)}')
        print(f'Test set size: {DataGenerator.ds_size_test()}')

    @staticmethod
    def _downsampling(
        args,
        input_layer, 
        filter_size, 
        dropout, 
        max_pool
        ):
        conv = tf.keras.layers.Conv2D(filter_size, kernel_size = 3, padding = 'same', activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(input_layer)
        conv = tf.keras.layers.Conv2D(filter_size, kernel_size = 3, padding = 'same', activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(conv)
        
        if dropout:
            conv = tf.keras.layers.Dropout(args.dropout)(conv)

        pool = tf.keras.layers.Layer()
        if max_pool:
            pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)

        return conv, pool

    @staticmethod
    def _upsampling_ConvTranspose(
        args,
        input_layer, 
        conv_layer, 
        filter_size
        ):
        X = tf.keras.layers.Conv2DTranspose(filter_size, kernel_size = 3, strides = 2, padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(input_layer)
        X = tf.keras.layers.Concatenate()([X, conv_layer])
        X = tf.keras.layers.Conv2D(filter_size, kernel_size = 3, padding = "same", activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(X)
        X = tf.keras.layers.Conv2D(filter_size, kernel_size = 3, padding = "same", activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(X)
        
        return X
    
    @staticmethod
    def _upsampling_UpSample(
        args,
        input_layer, 
        conv_layer, 
        filter_size
        ):
        X = tf.keras.layers.UpSampling2D(size = (2,2))(input_layer)
        X = tf.keras.layers.Conv2D(filter_size, kernel_size = 2, padding = "same", activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(X)
        X = tf.keras.layers.Concatenate()([X, conv_layer])
        X = tf.keras.layers.Conv2D(filter_size, kernel_size = 3, padding = "same", activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(X)
        X = tf.keras.layers.Conv2D(filter_size, kernel_size = 3, padding = "same", activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(X)
        
        return X

    @staticmethod
    def _build_model(
        args
        ):
        inputs = tf.keras.layers.Input((args.mask_size, args.mask_size, 1))

        conv1, pool1 = UnetModel._downsampling(args, input_layer = inputs, filter_size = args.max_filter_size / 16, dropout = False, max_pool = True)
        conv2, pool2 = UnetModel._downsampling(args, input_layer = pool1, filter_size = args.max_filter_size / 8, dropout = False, max_pool = True)
        conv3, pool3 = UnetModel._downsampling(args, input_layer = pool2, filter_size = args.max_filter_size / 4, dropout = False, max_pool = True)
        conv4, pool4 = UnetModel._downsampling(args, input_layer = pool3, filter_size = args.max_filter_size / 2, dropout = True, max_pool = True)
        conv5, _ = UnetModel._downsampling(args, input_layer = pool4, filter_size = args.max_filter_size, dropout = True, max_pool = False)

        if args.upsampling_method == 'trs':

            X = UnetModel._upsampling_ConvTranspose(args, conv5, conv4, args.max_filter_size / 2 )
            X = UnetModel._upsampling_ConvTranspose(args, X, conv3, args.max_filter_size / 4 )
            X = UnetModel._upsampling_ConvTranspose(args, X, conv2, args.max_filter_size / 8 )
            X = UnetModel._upsampling_ConvTranspose(args, X, conv1, args.max_filter_size / 16 )

        else:

            X = UnetModel._upsampling_UpSample(args, conv5, conv4, args.max_filter_size / 2 )
            X = UnetModel._upsampling_UpSample(args, X, conv3, args.max_filter_size / 4 )
            X = UnetModel._upsampling_UpSample(args, X, conv2, args.max_filter_size / 8 )
            X = UnetModel._upsampling_UpSample(args, X, conv1, args.max_filter_size / 16 )

        X = tf.keras.layers.Conv2D(2, kernel_size = 3, padding = "same", activation = tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l2(args.l2), kernel_initializer = 'he_normal')(X)
        X = tf.keras.layers.Conv2D(1, kernel_size = 1, padding = "same", activation = tf.nn.sigmoid)(X)

        model = tf.keras.Model(inputs = inputs, outputs = X)

        model.compile(
            optimizer =tf.keras.optimizers.Adam(lr = args.base_lr), 
            loss = loss_functions.soft_dice_loss,
            metrics = METRICS
            )
        
        return model
    
    def create_model(
        self, args
        ):
        self.model = UnetModel._build_model(args)
    
    def load_best_model(
        self, args, load_dir
        ):
        self.model = UnetModel._build_model(args)
        self.model.load_weights(load_dir)
        self.history = pickle.load(open(load_dir, "rb")) #open(os.path.join(load_dir, 'history')
        
    def train(
        self, args
        ):

        print('Training started:')
        history = self.model.fit(
            self.wf_train,
            epochs = args.n_epochs,
            steps_per_epoch = self.steps_per_epoch,
            validation_data = self.wf_dev,
            callbacks = CALLBACKS
            )
        self.history = history.history
        with open(args.savedir, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def evaluate(
        self, args
        ):
        print('Evaluating on train set:')
        loss, sdc, hdc, iou, acc = self.model.evaluate(self.wf_train, steps = self.steps_per_epoch, verbose=0)
        print(f'Train loss: {loss:.5f} - Train sdc: {sdc:.5f} - Train hdc: {hdc:.5f} - Train iou: {iou:.5f} - Train acc: {acc:.5f}')

        print('\nEvaluating on dev set:')
        loss, sdc, hdc, iou, acc = self.model.evaluate(self.wf_dev, verbose=0)
        print(f'Dev loss: {loss:.5f} - Dev sdc: {sdc:.5f} - Dev hdc: {hdc:.5f} - Dev iou: {iou:.5f} - Dev acc: {acc:.5f}')

        print('\nEvaluating on test set:')
        loss, sdc, hdc, iou, acc = self.model.evaluate(self.wf_test, verbose=0)
        print(f'Test loss: {loss:.5f} - Test sdc: {sdc:.5f} - Test hdc: {hdc:.5f} - Test iou: {iou:.5f} - Test acc: {acc:.5f}')

    def model_summary(
        self
        ):
        return self.model.summary()

    def get_trainable_params(
        self
        ):
        self.trainable_params = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
    




    




            
    