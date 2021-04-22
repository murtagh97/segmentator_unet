import tensorflow as tf

from arg_parser import args
from utils import loss_functions

METRICS = [
    loss_functions.soft_dice_coef, 
    loss_functions.hard_dice_coef, 
    loss_functions.iou,
    tf.keras.metrics.Accuracy()
    ]

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir = args.logdir, 
    histogram_freq = 1, 
    update_freq = 100, 
    profile_batch =0 
    )

lr_decay_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.5, 
    patience = 4, 
    min_lr = args.min_lr,
    verbose = 1 
    )

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 40,
    verbose = 1,
    )

checkpointing_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = args.savedir, 
    monitor = 'val_loss', 
    save_freq = 'epoch',
    save_best_only = True,
    save_weights_only = True,
    verbose=1,
    )

CALLBACKS = [
    tb_callback,
    lr_decay_callback,
    checkpointing_callback
    ]

