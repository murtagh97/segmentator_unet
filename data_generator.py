import os
import glob

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

class DataGenerator:

    @staticmethod
    def _sort_filenames(
        fold, side, 
        file_end
        ):
        """Alphabetically sort the filenames in given directory.

        Parameters
          - work_dir: Working directory that contains folder with the data, i.e., lungs_data folder.
          - arg_set: String argument, either fold1 or fold2, where fold1 stores train data, fold2 stores test data. 
          - arg_side: String argument, either right or left, where fold1 stores train data, fold2 stores test data. 
          - arg_file: String argument, either gif, for masks, or bmp, for images.
               
        Returns
          - Calculated value of the Soft Dice Coefficient.
        """
        work_dir = os.getcwd()
        data_dir = os.path.join(work_dir, 'lungs_data', fold, side)
        filenames = glob.glob(data_dir + "/*." + file_end)
        filenames.sort()

        return filenames

    @staticmethod
    def _get_masks_from_dir(
        fold
        ):
        """Alphabetically sort the filenames in given directory.

        Parameters
          - work_dir: Working directory that contains folder with the data, i.e., lungs data folder.
          - arg_set: String argument, either fold1 or fold2, where fold1 contains the train data, fold2 contains the test data.  
              
        Returns
          - Calculated value of the Soft Dice Coefficient.
        """

        filenames_left = DataGenerator._sort_filenames(fold, 'left', 'gif')
        filenames_right = DataGenerator._sort_filenames(fold, 'right', 'gif') 

        res_list = []
        for (left,right) in zip(filenames_left, filenames_right):
            lung_left = plt.imread(left)
            lung_right = plt.imread(right)
            lungs_both = lung_left + lung_right
            res_list.append(lungs_both)

        res_np = np.array(res_list)

        # expand dimensions, add third channel representing color
        res_fin = np.expand_dims(np.asarray(res_np, dtype = np.float), axis = 3)

        return res_fin

    @staticmethod
    def _get_images_from_dir(
        fold
        ):
        """Alphabetically sort the filenames in given directory.

        Parameters
          - work_dir: Working directory that contains folder with the data, i.e., lungs data folder.
          - arg_set: String argument, either fold1 or fold2, where fold1 contains the train data, fold2 contains the test data.  
              
        Returns
          - Calculated value of the Soft Dice Coefficient.
        """

        filenames = DataGenerator._sort_filenames(fold, 'images', 'bmp') 

        res_list = []
        for name in filenames:
            lungs_image = plt.imread(name)
            res_list.append(lungs_image[:,:,0])

        res_np = np.array(res_list)

        # expand dimensions, add third channel representing color
        res_fin = np.expand_dims(np.asarray(res_np), axis = 3)

        return res_fin
    
    ### - data normalization function - ###
    @staticmethod
    def normalize_ds(
        image, mask, 
        mask_size
        ):
        image = ( tf.cast(image, tf.float32) / 255.0)
        
        mask = tf.image.resize(mask, [mask_size, mask_size])
        mask = ( tf.cast(mask, tf.float32) / 255.0)

        # binarize input mask
        mask = tf.cast(mask >= 0.5, tf.float32)
        
        return image, mask

    ### - randomly flip both image and mask horizontally (left to right) - ###
    @staticmethod
    def rand_flip_lr(
        image, mask
        ):
        flip_cond = tf.random.uniform([], 0, 1.0) # flip condition
        
        if flip_cond >= 0.5:
          image = tf.image.flip_left_right(image)
          mask = tf.image.flip_left_right(mask)  
        
        return image, mask

    ### - randomly center crop both image and mask and pad them back to original shape, i.e does "zooming" (larger arg_crop, smaller "zoom") - ###
    @staticmethod
    def rand_central_crop(
        image, mask, 
        mask_size, crop
        ):
        crop_cond = tf.random.uniform([], 0, 1.0) # crop condition
        
        if crop_cond >= 0.5:
          image = tf.image.central_crop(image, central_fraction = crop)
          mask = tf.image.central_crop(mask, central_fraction = crop)

          image = tf.image.resize_with_pad(image, mask_size, mask_size)
          mask = tf.image.resize_with_pad(mask, mask_size, mask_size)

        return image, mask

    ### - randomly adjust brigthness for image (larger delta, brighter) - ###
    @staticmethod
    def rand_brightness(
        image, mask, 
        bright
        ):
        bright_cond = tf.random.uniform([], 0, 1.0) # adjustment condition
        
        if bright_cond >= 0.5:
          image = tf.image.adjust_brightness(image, bright)
          image = tf.clip_by_value(image, 0.0, 1.0)

        return image, mask

    ### - randomly rotation of both image and mask - ###
    @staticmethod
    def rand_rotation(
        image, mask, 
        rot_angle
        ):
        rotation_condition = tf.random.uniform([], 0, 1.0) # rotation condition
        rotation_direction = tf.random.uniform([], 0, 1.0) # rotate clockwise / counterclockwise condition

        if rotation_condition >= 0.5:
          
          rand_angle = np.random.uniform(0, rot_angle)
          if rotation_direction >= 0.5:

            image = tfa.image.rotate(image, rand_angle, interpolation = 'nearest', fill_mode = 'reflect')
            mask = tfa.image.rotate(mask, rand_angle, interpolation = 'nearest', fill_mode = 'reflect')

          else:

            image = tfa.image.rotate(image, -rand_angle, interpolation = 'nearest', fill_mode = 'reflect')
            mask = tfa.image.rotate(mask, -rand_angle, interpolation = 'nearest', fill_mode = 'reflect')

        return image, mask

    @staticmethod
    def augmentation(
        image, mask,
        mask_size,
        flip, 
        crop, 
        bright,
        rot_angle, 
        ):
        image, mask = DataGenerator.normalize_ds(image, mask, mask_size)

        if flip:
            image, mask = DataGenerator.rand_flip_lr(image, mask)
        
        if crop > 0:
            image, mask = DataGenerator.rand_central_crop(image, mask, mask_size, crop)
        
        if bright > 0:
            image, mask = DataGenerator.rand_brightness(image, mask, bright)
        
        if rot_angle > 0:
            image, mask = DataGenerator.rand_rotation(image, mask, rot_angle)
        
        return image, mask

    @staticmethod
    def get_tf_wf(
        train_dev_split, 
        shuffle_size, 
        epochs,
        batch_size, 
        mask_size, 
        flip, 
        crop, 
        bright,
        rot_angle,  
        random_seed
        ):

        masks_train = DataGenerator._get_masks_from_dir('fold1')
        masks_test = DataGenerator._get_masks_from_dir('fold2')

        images_train = DataGenerator._get_images_from_dir('fold1')
        images_test = DataGenerator._get_images_from_dir('fold2')

        ds_train_size = DataGenerator.ds_size_train(train_dev_split)
        ds_dev_size = DataGenerator.ds_size_dev(train_dev_split)

        # - create tf.data datasets from np.arrays -
        ds_train = tf.data.Dataset.from_tensor_slices((images_train, masks_train))
        ds_test = tf.data.Dataset.from_tensor_slices((images_test, masks_test))

        # - shuffle ds_train before performing train-dev split - #
        ds_train = ds_train.shuffle(shuffle_size, seed = random_seed)
        
        # - create validation split of train_ds -
        wf_train = ds_train.take(ds_train_size)
        wf_dev = ds_train.skip(ds_train_size)

        # - create train and test workflows -
        # create train wf by taking first N samples from (shuffled) train_ds -> shuffle -> augment -> batch
        wf_train = wf_train.shuffle(shuffle_size, seed = random_seed)
        wf_train = wf_train.map(
            lambda image, mask: DataGenerator.augmentation(
                image, mask, 
                mask_size,
                flip,
                crop,
                rot_angle,
                bright
                )
            )

        wf_train = wf_train.repeat()
        wf_train = wf_train.batch(batch_size)
        wf_train = wf_train.prefetch(tf.data.experimental.AUTOTUNE)

        # create val wf by skipping first N samples from (shuffled) train_ds -> normalize -> batch
        wf_dev = wf_dev.map(
            lambda image, mask: DataGenerator.normalize_ds(
                image, mask, 
                mask_size
                )
            )
        wf_dev = wf_dev.batch(batch_size)
        wf_dev = wf_dev.prefetch(tf.data.experimental.AUTOTUNE)

        # create test wf from test_ds : normalize -> batch
        wf_test = ds_test.map(
            lambda image, mask: DataGenerator.normalize_ds(
                image, mask, 
                mask_size
                )
            )
        wf_test = wf_test.batch(batch_size)
        wf_test = wf_test.prefetch(tf.data.experimental.AUTOTUNE)

        return wf_train, wf_dev, wf_test   

    @staticmethod
    def ds_size_train(train_dev_split):
        work_dir = os.getcwd()
        data_dir = os.path.join(work_dir, 'lungs_data', 'fold1', 'left')

        n_files = len(os.listdir(data_dir))

        size = int(train_dev_split * n_files)
        
        return size
    
    @staticmethod
    def ds_size_dev(train_dev_split):
        work_dir = os.getcwd()
        data_dir = os.path.join(work_dir, 'lungs_data', 'fold1', 'left')

        n_files = len(os.listdir(data_dir))

        train_size = int(train_dev_split * n_files)
        dev_size = n_files - train_size
        
        return dev_size

    @staticmethod
    def ds_size_test():
        work_dir = os.getcwd()
        data_dir = os.path.join(work_dir, 'lungs_data', 'fold2', 'left')

        size = len(os.listdir(data_dir))
        
        return size

  


  