from __future__ import division
import os,sys
sys.path.append('../keras')
import math
import time
from model_config import *
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os, sys
from scipy import misc
from skimage.color import rgb2gray
import numpy as np
from utilities import preprocess_images, preprocess_maps, postprocess_predictions
from keras import backend as K
from model import ml_net_model, loss
from keras.models import load_model


def generator(b_s, images, maps, shape_r, shape_c, shape_r_gt, shape_c_gt,shuffle=True):
    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r_gt, shape_c_gt)
        counter = (counter + b_s)

def generator_test(b_s, images):
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

def finetune_model(prog_path,nb_epoch,train_iters,val_iters,training_image_path,training_map_path,weight_path,output_folder,test_images,test_output,shuffle=True):
    #config settings
    imgs_train_path = training_image_path
    maps_train_path = training_map_path
    #nb_imgs_train = train_iters
    #imgs_val_path = training_image_path
    #maps_val_path = training_map_path
    #nb_imgs_val = val_iters
    #b_s, shape_r, shape_c, shape_r_gt, shape_c_gt, _= config()

    #start finetunings
    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10,weight_path=weight_path)
    datagen = ImageDataGenerator()
    sgd = SGD(lr=1e-4, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model for finetuning")
    model.compile(sgd, loss)
    print("Load weights ML-Net")
    model.load_weights(weight_path + '/mlnet_salicon_weights.pkl')
    model = prepare_finetune(model,4,sgd)

    #prepare model
    timestamp = time.localtime()
    timestamp = prog_path + 'checkpoints/'  + str(timestamp.tm_mon) + '_' + str(timestamp.tm_mday) + '_' + str(timestamp.tm_hour) + '_' + str(timestamp.tm_min)
    if not os.path.exists(timestamp):
        os.makedirs(timestamp)
    print('Finetuning model for ' + str(nb_epoch) + ' epochs')
    for ep in range(nb_epoch):
	ep_loss = 0
	num_batches = 0
        batch_estimate = int(np.ceil(len(imgs_train_path) / b_s))
        if shuffle:
            rand_order = np.arange(len(imgs_train_path))
            np.random.shuffle(rand_order)
            ar_images = np.asarray(imgs_train_path)
            ar_maps = np.asarray(maps_train_path)
            ar_images = ar_images[rand_order]
            ar_maps = ar_maps[rand_order]
            it_imgs_train_path = ar_images.tolist()
            it_maps_train_path = ar_maps.tolist()
        else:
            it_imgs_train_path = np.copy(it_imgs_train_path).tolist()
            it_maps_train_path = np.copy(it_maps_train_path).tolist()

        for X_train, Y_train in generator(b_s, it_imgs_train_path, it_maps_train_path, shape_r, shape_c, shape_r_gt, shape_c_gt):
            if X_train.shape[0] == 0:
                break
            else:
                ep_loss += model.train_on_batch(X_train, Y_train)
                num_batches += 1
	        sys.stdout.write('\r' + str(num_batches) + '/' + str(batch_estimate))
                sys.stdout.flush()
        print('mean loss across batches', ep_loss / num_batches)
        model_pointer = timestamp + '/' + str(ep) + '.h5'
        model.save(model_pointer)

    #Now make predictions
    prediction_paths = make_predictions(model,test_images,test_output)
    return model_pointer, prediction_paths

def make_predictions(model,imgs_test_path,output_folder):
    print("Predict saliency maps for " + imgs_test_path)
    predictions = model.predict_generator(generator_test(b_s=1, imgs_test_path=imgs_test_path), nb_imgs_test)
    prediction_paths = []
    for pred, name in zip(predictions, imgs_test_path):
        original_image = misc.imread(name)
        if len(original_image.shape) > 2:
            original_image = rgb2gray(original_image)
        res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        im_name = re.split('/',imgs_test_path)[-1]
        pred_name = output_folder + im_name
        misc.imsave(pred_name, res.astype(int))
        prediction_paths.append(pred_name)
    return prediction_paths

def produce_maps(weight_path, checkpoint_path, imgs_test_path, output_folder):
    #model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10,weight_path=weight_path)
    #sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    #print("Compile ML-Net Model")
    #model.compile(sgd, loss)

    # path of output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nb_imgs_test = len(imgs_test_path)

    print("Load weights ML-Net")
    model = load_model(checkpoint_path) #loading from the finetuned model
    #model.load_weights(checkpoint_path)

    print("Predict saliency maps for " + imgs_test_path)
    predictions = model.predict_generator(generator_test(b_s=1, imgs_test_path=imgs_test_path), nb_imgs_test)
    prediction_paths = []
    for pred, name in zip(predictions, imgs_test_path):
        original_image = misc.imread(name)
        if len(original_image.shape) > 2:
            original_image = rgb2gray(original_image)
        res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        im_name = re.split('/',imgs_test_path)[-1]
        pred_name = output_folder + im_name
        misc.imsave(pred_name, res.astype(int))
        prediction_paths.append(pred_name)
    return prediction_paths
