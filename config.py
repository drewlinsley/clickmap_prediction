import math

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# batch size
b_s = 10
# number of rows of input images
shape_r = 480
# number of cols of input images
shape_c = 640
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r / 8))
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c / 8))
# number of epochs
nb_epoch = 20

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training mages
imgs_train_path = 'MIRC_data/images/'
# path of training maps
maps_train_path = 'MIRC_data/click_maps/'
# number of training images
nb_imgs_train = 10000
# path of validation images
imgs_val_path = 'MIRC_data/images/'
# path of validation maps
maps_val_path = 'MIRC_data/click_maps/'
# number of validation images
nb_imgs_val = 110
