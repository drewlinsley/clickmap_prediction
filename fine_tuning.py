from __future__ import division
import sys
sys.path.append('../keras')
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os, sys
from scipy import misc
from skimage.color import rgb2gray
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, postprocess_predictions
from model import ml_net_model, loss
from keras import backend as K


def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith('.JPEG')]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith('.JPEG')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.JPEG')]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith('.JPEG')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()

    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r_gt, shape_c_gt)
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith('.JPEG')]
    images.sort()
    counter = 0
    while True:
        im = preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        yield im
        counter = (counter + b_s) % len(images)

def remove_prior(model,num_pop):
    for idx in range(num_pop):
        model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    return model

def prepare_finetune(model,num_finetunes,sgd):
    #pop unnecessary layers here
    for layer in model.layers[:len(model.layers)-num_finetunes]:
        layer.trainable=False
    model = reset_model(model) #and initialize the trainable weights
    model.compile(sgd, loss) #have to recompile
    return model

def reset_model(model):
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'init') and layer.trainable == True:
            init = getattr(layer, 'init')
            new_weights = init(layer.get_weights()[0].shape).get_value()
            #bias = K.zeros(shape=(layer.get_weights()[1].shape)).get_value()
            if len(layer.get_weights()) > 1:
                bias = np.zeros((layer.get_weights()[1].shape)).astype(np.float32)
                layer.set_weights([new_weights, bias])
            else:
                layer.set_weights([new_weights])
            model.layers[idx] = layer
    return model

def finetune_model(input_images,weight_path,output_folder):
    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10,weight_path=weight_path)
    datagen = ImageDataGenerator()
    sgd = SGD(lr=1e-4, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model for finetuning")
    model.compile(sgd, loss)
    print("Load weights ML-Net")
    print(weight_path)
    model.load_weights(weight_path + '/mlnet_salicon_weights.pkl')
    model = prepare_finetune(model,4,sgd)
    import ipdb;ipdb.set_trace()
    for X_train, Y_train in generator(b_s=b_s):
        for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32): # these are chunks of 32 samples
            it_loss = model.train_on_batch(X_batch, Y_batch)
            print(it_loss)
            #Figure out how to checkpoint the model here

def produce_maps(imgs_test_path, weight_path,output_folder):
    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(sgd, loss)
    # path of output folder
    file_names = [f for f in os.listdir(imgs_test_path) if f.endswith('.JPEG')]
    nb_imgs_test = len(file_names)

    print("Load weights ML-Net")
    #Look into the database for the current model checkpoint
    model.load_weights(weight_path)

    print("Predict saliency maps for " + imgs_test_path)
    predictions = model.predict_generator(generator_test(b_s=1, imgs_test_path=imgs_test_path), nb_imgs_test)
    for pred, name in zip(predictions, file_names):
        original_image = misc.imread(imgs_test_path + name)
        if len(original_image.shape) > 2:
            original_image = rgb2gray(original_image)
        res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        misc.imsave(output_folder + '%s' % name, res.astype(int))
