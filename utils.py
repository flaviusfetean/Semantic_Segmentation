from tensorflow.keras.utils import to_categorical
import numpy as np


def one_hot_encoding(y, num_classes):

    encs = []
    for i in range(num_classes):
        encs.append(to_categorical(i, num_classes))

    image_shape = y[0].shape

    seg_labels = np.ndarray((image_shape[0], image_shape[1], num_classes), dtype=np.uint8)
    seg_labeled_imgs = []
    for data_point in y:
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                seg_labels[i, j] = encs[data_point[i, j, 2]]
        seg_labeled_imgs.append(seg_labels)

    return np.array(seg_labeled_imgs)


def convert_one_hot_to_rgb(prediction, palette):
    seg_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    map_img = np.zeros((prediction.shape[0], prediction.shape[1]), dtype=np.uint8)

    prediction = np.argmax(prediction, axis=-1)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            color_label = np.argmax(prediction[i, j])
            map_img = color_label
            for k in range(3):
                seg_img[i, j, k] = palette[color_label][k]

    return seg_img, map_img
