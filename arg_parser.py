import os
import datetime
import argparse

parser = argparse.ArgumentParser(
    description = 'Segmentation arguments'
    )
parser.add_argument('-f')# dummy argument for ipynb
parser.add_argument(
    "--train_dev_split", 
    default = 0.85, 
    type = float,
    help= "Train-dev split ratio."
    )
parser.add_argument(
    "--batch_size", 
    default = 8, 
    type = int,
    help= "Batch size."
    )
parser.add_argument(
    "--n_epochs", 
    default = 175, 
    type = int,
    help = "Number of epochs."
    )    
parser.add_argument(
    "--mask_size", 
    default = 256,
    type = int,
    help = "Mask size."
    )
parser.add_argument(
    "--flip", 
    default = False, 
    type = bool, 
    help = "Random flip left-right."
    )
parser.add_argument(
    "--crop", 
    default = 0.7, 
    type = float, 
    help = "Crop size."
    )
parser.add_argument(
    "--bright", 
    default = 0.15, 
    type = float, 
    help = "Brightness delta."
    )
parser.add_argument(
    "--rot_angle", 
    default = 0, 
    type = float, 
    help = "Rotation angle size."
    )
parser.add_argument(
    "--upsampling_method", 
    default = "trs",
    type = str,
    help = "Upsample method: either trs or ups."
    )
parser.add_argument(
    "--max_filter_size", 
    default = 512,
    type = int,
    help = "Largest number of filters in the Unet model."
    )
parser.add_argument(
    "--l2", 
    default = 10e-5, 
    type = float, 
    help = "L2 regularization."
    )
parser.add_argument(
    "--dropout", 
    default = 0.5, 
    type = float, 
    help = "Dropout."
    )
parser.add_argument(
    "--base_lr", 
    default = 10e-5, 
    type = float, 
    help = "Base learning rate."
    )
parser.add_argument(
    "--min_lr", 
    default = 10e-13, 
    type = int, 
    help = "Base learning rate."
    )
parser.add_argument(
    "--shuffle", 
    default = 1000, 
    type = int, 
    help = "Shuffle size."
    )
parser.add_argument(
    "--random_seed", 
    default = 22, 
    type = int, 
    help = "Random seed."
    )
parser.add_argument(
    "--plot_model", 
    default = False, 
    type = bool, 
    help = "Plot model architecture."
    )
parser.add_argument(
    "--print_model", 
    default = True, 
    type = bool, 
    help = "Print model architecture."
    )

args = parser.parse_args()

logdir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") 
logdir_name += '_bs-' + str(args.batch_size) + '_epochs-' + str(args.n_epochs) + '_baselr-' + str(args.base_lr)
logdir_name += '_dropout-' + str(args.dropout) + '_l2-' + str(args.l2) 
logdir_name += '_flip-' + str(args.flip) + '_crop-' + str(args.crop) + '_bright-' + str(args.bright) + '_rot-' + str(args.rot_angle)

args.logdir = os.path.join( "logs", logdir_name)
args.savedir = os.path.join( args.logdir, 'saved_model', 'unet_best_model',)