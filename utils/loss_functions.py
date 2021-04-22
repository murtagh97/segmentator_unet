import tensorflow as tf

from arg_parser import args

def soft_dice_coef(
    y_true, y_pred, 
    smooth = 1.
    ):
    """Dice Coefficient computed directly from the predicted probabilities, i.e., the so-called Soft Dice Coefficient.

    Parameters
      - y_true: Ground truth mask.
      - y_pred: Predicted mask, output of the network. 
      - smooth: Smoothing constant added for mathematical stability to avoid division by zero. By default smooth = 1.
          
    Returns
      - Calculated value of the Soft Dice Coefficient.

    References
      - https://en.wikipedia.org/wiki/Dice_coefficient.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def hard_dice_coef(
    y_true, y_pred, 
    mask_size = args.mask_size, 
    smooth = 1.
    ):
    """Dice Coefficient computed from the thresholded predicted probabilities, i.e., the so-called Hard Dice Coefficient.

    Parameters
      - y_true: Ground truth mask.
      - y_pred: Binary predicted mask, thresholded output of the network at the threshold value of 0.5. 
      - smooth: Smoothing constant added for mathematical stability to avoid division by zero. By default smooth = 1.
          
    Returns
      - Calculated value of the Hard Dice Coefficient.

    References
      - https://en.wikipedia.org/wiki/Dice_coefficient.
    """
    y_true_f = tf.reshape(tf.cast(y_true >= 0.5, tf.float32), [-1, mask_size * mask_size])
    y_pred_f = tf.reshape(tf.cast(y_pred >= 0.5, tf.float32), [-1, mask_size * mask_size])

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou(
    y_true, y_pred, 
    mask_size = args.mask_size
    ):
    """Intersection over Union computed from the thresholded predicted probabilities.

    Parameters
      - y_true: Ground truth mask.
      - y_pred: Binary predicted mask, thresholded output of the network at the threshold value of 0.5. 
          
    Returns
      - Calculated value of the Intersetion over Union.

    References 
      - https://en.wikipedia.org/wiki/Jaccard_index.
    """
    y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, mask_size * mask_size])
    y_pred_mask = tf.reshape(tf.math.round(y_pred) == 1, [-1, mask_size * mask_size])
  
    intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
    union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

    intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
    union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

    return tf.where(union == 0, 1., intersection / union)

def soft_dice_loss(
    y_true, y_pred
    ):
    """Dice loss function computed from the Soft Dice Coefficient.

    Parameters
      - y_true: Ground truth mask.
      - y_pred: Predicted mask, output of the network. 
          
    Returns
      - Calculated value of the Dice Loss.

    References
      - https://en.wikipedia.org/wiki/Dice_coefficient.
    """

    return 1 - soft_dice_coef(y_true, y_pred)

def binary_ce_soft_dice_loss(
    y_true, y_pred
    ):
    """Loss function combining the binary cross-entropy and the soft dice loss.

    Parameters
      - y_true: Ground truth mask.
      - y_pred: Predicted mask, output of the network. 
          
    Returns
      - Calculated value of the combined loss.

    References
      - https://en.wikipedia.org/wiki/Dice_coefficient.
      - https://en.wikipedia.org/wiki/Cross_entropy.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    return bce(y_true, y_pred) + soft_dice_loss(y_true, y_pred)