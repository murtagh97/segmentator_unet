import numpy as np
import tensorflow as tf

from arg_parser import args
from model_object import UnetModel

def main(args):
    
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    unet_model = UnetModel(args) 

    unet_model.prepare_data(args)

    unet_model.create_model(args)

    unet_model.train(args)

    unet_model.load_best_model(args, load_dir= args.savedir)

    unet_model.evaluate(args)

if __name__ == "__main__":
     main(args)