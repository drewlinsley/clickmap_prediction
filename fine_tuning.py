from __future__ import division
import os
import sys
import re
import math
import numpy as np
from scipy import misc
from skimage.color import rgb2gray
sys.path.append('../keras')
from keras.optimizers import SGD, Adam
from utilities import preprocess_images, preprocess_maps, postprocess_predictions
from clicktionary_model import ml_net_model, attention_loss
#from model import ml_net_model, loss


def generator(b_s, images, maps, shape_r, shape_c, shape_r_gt, shape_c_gt, augmentations=None):
    counter = 0
    while True:
        im_batch = images[counter:counter + b_s]
        map_batch = maps[counter:counter + b_s]
        augmentation_index = np.random.rand(len(im_batch)) > .5  #uniform random augmentations
        yield preprocess_images(im_batch, shape_r, shape_c, augmentation_index, augmentations=augmentations),\
            preprocess_maps(map_batch, shape_r_gt, shape_c_gt, augmentation_index, augmentations=augmentations)
        counter = (counter + b_s)


def generator_test(b_s, images, shape_r, shape_c):
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


def prepare_finetune(model,shape_r_gt, shape_c_gt, num_finetunes, sgd):
    """pop unnecessary layers and initialize the trainable weights here"""
    for layer in model.layers[:len(model.layers)-num_finetunes]:
        layer.trainable=False
    model = reset_model(model)
    model.compile(sgd, loss=attention_loss(shape_r_gt=shape_r_gt,shape_c_gt=shape_c_gt))
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


def finetune_model(p, shape_r_gt, shape_c_gt,\
    training_image_path, training_map_path, shuffle=True,\
    initialize_with_mlnet=False, optimizer='sgd'):

    model = ml_net_model(img_cols=p.model_input_shape_c,
        img_rows=p.model_input_shape_r, downsampling_factor_product=10,
        weight_path=p.model_init_training_weights)

    print("Compile ML-Net Model for finetuning")
    shape_r_gt = int(math.ceil(p.model_input_shape_r / 8))
    shape_c_gt = int(math.ceil(p.model_input_shape_c / 8))
    if optimizer == 'sgd':
        optim = SGD(lr=1e-4, decay=0.0005, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        optim = Adam(lr=3e-4)
    model.compile(optim, loss=attention_loss(shape_r_gt=shape_r_gt,
        shape_c_gt=shape_c_gt))

    if initialize_with_mlnet:
        print("Load weights ML-Net")
        model.load_weights(
            os.path.join(p.model_init_training_weights,
            '/mlnet_salicon_weights.pkl'))  # This is questionable
        model = prepare_finetune(model, 3, optim)  # 3 should finetune the layers devoted to eye gaze prediction

    # prepare model
    timestamp = os.path.join(p.model_path, p.model_checkpoints, p.dt_string)
    if not os.path.exists(timestamp):
        os.makedirs(timestamp)

    print('Finetuning model for ' + str(p.nb_epoch) + ' epochs')
    for ep in range(p.nb_epoch):
        ep_loss = 0
        prev_loss = 0
        num_batches = 0
        batch_estimate = int(np.ceil(len(training_image_path) / p.batch_size))
        if shuffle:
            rand_order = np.arange(len(training_image_path))
            np.random.shuffle(rand_order)
            ar_images = np.asarray(training_image_path)
            ar_maps = np.asarray(training_map_path)
            ar_images = ar_images[rand_order]
            ar_maps = ar_maps[rand_order]
            it_imgs_train_path = ar_images.tolist()
            it_maps_train_path = ar_maps.tolist()
        else:
            it_imgs_train_path = np.copy(it_imgs_train_path).tolist()
            it_maps_train_path = np.copy(it_maps_train_path).tolist()

        for X_train, Y_train in generator(p.batch_size, it_imgs_train_path,
            it_maps_train_path, p.model_input_shape_r, p.model_input_shape_c,
            shape_r_gt, shape_c_gt, augmentations=p.augmentations):
            if X_train.shape[0] == 0:
                break
            else:
                ep_loss += model.train_on_batch(X_train, Y_train)
                num_batches += 1
            sys.stdout.write('\r' + str(num_batches) + '/' + str(batch_estimate) +
                ' | ' + 'Batch loss delta is: ' + str(ep_loss - prev_loss))
            prev_loss = ep_loss
            sys.stdout.flush()
        print ' || mean loss across batches is %0.5f' % (ep_loss / num_batches)
        model_pointer = os.path.join(timestamp, str(ep) + '.h5')
        model.save(model_pointer)

    # Now make predictions
    # prediction_paths = make_predictions(model,test_images,test_output)
    return model_pointer  # , prediction_paths

def make_predictions(model, imgs_test_path, output_folder):
    nb_imgs_test = len(imgs_test_path)
    predictions = model.predict_generator(generator_test(1, imgs_test_path), nb_imgs_test)
    prediction_paths = []
    for pred, name in zip(predictions, imgs_test_path):
        original_image = misc.imread(name)
        if len(original_image.shape) > 2:
            original_image = rgb2gray(original_image)

        res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        im_name = re.split('/', imgs_test_path)[-1]
        pred_name = os.path.join(output_folder,im_name)
        misc.imsave(pred_name, res.astype(int))
        prediction_paths.append(pred_name)
    return prediction_paths

def produce_maps(p, checkpoint_path, imgs_test_path):

    nb_imgs_test = len(imgs_test_path)

    print("Load weights ML-Net")
    model = ml_net_model(img_cols=p.model_input_shape_c, img_rows=p.model_input_shape_r,
        downsampling_factor_product=10, weight_path=p.model_init_training_weights)
    model.load_weights(checkpoint_path)
    # model = load_model(checkpoint_path) #loading from the finetuned model
    # model.load_weights(checkpoint_path)
    predictions = model.predict_generator(
        generator_test(b_s=1, images=imgs_test_path,
            shape_r=p.model_input_shape_r, shape_c=p.model_input_shape_c),
        nb_imgs_test)
    prediction_paths = []
    for pred, name in zip(predictions, imgs_test_path):
        original_image = misc.imread(name)
        if len(original_image.shape) > 2:
            original_image = rgb2gray(original_image)
        res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        im_name = re.split('/', name)[-1]
        pred_name = os.path.join(p.prediction_output_path, im_name)
        misc.imsave(pred_name, res.astype(int))
        prediction_paths.append(pred_name)
    return prediction_paths
