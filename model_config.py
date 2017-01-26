import math

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# batch size
b_s = 24
# number of rows of input images
shape_r = 480 #NEED TO MOVE THIS TO THE OTHER CONFIGcd /
# number of cols of input images
shape_c = 640
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r / 8))
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c / 8))
# number of epochs
nb_epoch = 1
