from __future__ import division
import cv2
import numpy as np
from scipy import misc
from skimage.color import rgb2gray

def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = misc.imresize(img, [new_cols, shape_r])
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = misc.imresize(img, [shape_c, new_rows])
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def augment_image(image, augmentations):
    """Concatenate augmented versions on this batch.
    These are commutative so it's ok that they're in this order """

    if 'lr_flip' in augmentations:
        image = np.fliplr(np.squeeze(image))

    if 'noise' in augmentations:
        hw = image.shape
        if len(hw) > 2:
            noise_image = np.repeat(
                (np.random.rand(hw[0], hw[1])
                    * 2 - 1)[:, :, None], hw[2], axis=-1)
        else:
            noise_image = np.random.rand(hw[0],hw[1]) * 2 - 1
        image += noise_image

    return image


def preprocess_images(paths, shape_r, shape_c, augmentation_index, augmentations=None):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = misc.imread(path)
        if len(original_image.shape) < 3:
            original_image = np.repeat(original_image[: , :, None], 3, axis=2)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        if augmentations is not None:
            if augmentation_index[i]:
                padded_image = augment_image(padded_image,augmentations)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims.transpose((0, 3, 1, 2))
    return ims

def preprocess_h5_images(key, h5_file, shape_r, shape_c, augmentation_index, augmentations=None):
    batch_len = len(augmentation_index)
    ims = np.zeros((batch_len, shape_r, shape_c, 3))
    for i in range(batch_len):
        original_image = cv2.imdecode(h5_file[key][i], 1)
        if len(original_image.shape) < 3:
            original_image = np.repeat(original_image[: , :, None], 3, axis=2)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        if augmentations is not None:
            if augmentation_index[i]:
                padded_image = augment_image(padded_image,augmentations)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims.transpose((0, 3, 1, 2))
    return ims


def preprocess_maps(paths, shape_r, shape_c, augmentation_index, augmentations=None):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        original_map = misc.imread(path)
        if len(original_map.shape) > 2:
            original_map = rgb2gray(original_map)
        padded_map = padding(original_map, shape_r, shape_c, 1).astype(np.float32)
        if augmentations is not None:
            if augmentation_index[i]:
                padded_map = augment_image(padded_map, augmentations)
        ims[i, 0] = padded_map / 255.0

    return ims


def preprocess_h5_maps(key, h5_file, shape_r, shape_c, augmentation_index, augmentations=None):
    batch_len = len(augmentation_index)
    ims = np.zeros((batch_len, 1, shape_r, shape_c))
    for i in range(batch_len):
        original_map = cv2.imdecode(h5_file[key][i], 0)
        if len(original_map.shape) > 2:
            original_map = rgb2gray(original_map)
        padded_map = padding(original_map, shape_r, shape_c, 1).astype(np.float32)
        if augmentations is not None:
            if augmentation_index[i]:
                padded_map = augment_image(padded_map, augmentations)
        ims[i, 0] = padded_map / 255.0

    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = misc.imresize(pred, [new_cols, shape_r])
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = misc.imresize(pred, [shape_c, new_rows])
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / np.max(img) * 255
