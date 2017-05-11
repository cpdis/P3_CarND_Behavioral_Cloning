import os, time, math, random, cv2, json, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook

import keras
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dropout, Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from IPython.display import display
from pathlib import Path
import json

driving_data = pd.read_csv("driving_log.csv", index_col = False)
driving_data.columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']

# 1. Flip horizontally
def horizontal_flip(img, steering_angle):
    """
    Flips the image horizontally and corrects the steering angle

    Parameters
    ----------
    img             : image being passed in
    steering_angle  : steering angle corresponding to the image

    Returns
    -------
    The new image flipped horizontally
    """
    flipped = cv2.flip(img, 1) # positive (>0) flip code means flipping about y-axis
    steering_angle = steering_angle * -1 # change sign of steering angle to account for flip
    return flipped, steering_angle

# 2. Brighness shift
BRIGHT_VAL = 0.5
def brightness_shift(img):
    """
    Converts to HSV color space to change brightness (luminosity) and returns new image

    Parameters
    ----------
    img    : image being passed in

    Returns
    -------
    The new image with randomly augmented brightness
    """
    img_new = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # convert color spaces
    random_bright_val = BRIGHT_VAL + np.random.uniform() # random coefficient to shift brightness by
    img_new[:,:,2] = img_new[:,:,2] * random_bright_val
    img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2RGB) # convert back to RGB
    return img_new

# 3./4. Height and width shift
WIDTH_SHIFT = 100
HEIGHT_SHIFT = 50
def height_width_shift(img, steer, translate_range):
    """
    Shifts the image in the x and y directions based on specified height and width shifts

    Parameters
    ----------
    img             : image being passed in
    steer           : steering angle
    translate_range : maximum translation

    Returns
    -------
    The new image translated in the x and y directions
    """
    rows, columns, channels = img.shape

#     y
#     ^
#     |
#     |____> x

    translate_x = WIDTH_SHIFT * np.random.uniform() - WIDTH_SHIFT / 2
    translate_y = HEIGHT_SHIFT * np.random.uniform() - HEIGHT_SHIFT / 2
    transform = np.float32([[1, 0 , translate_x], [0, 1, translate_y]])
    steering_angle = steer + translate_x/translate_range * 2 * .2

    transformed_img = cv2.warpAffine(img, transform, (columns, rows)) # example warpAffine from http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    return transformed_img, steering_angle, translate_x

threshold = 1
img_rows = 64
img_columns = 64
img_channels = 3

# Preprocess image files
def preprocessImage(img):
    """
    Removes the irrelevant areas of the image (car hood and above horizon) and resizes the
    image to the specified dimensions

    Parameters
    ----------
    img             : image being passed in

    Returns
    -------
    The new image cropped and resized
    """
    shape = img.shape
    img = img[math.floor(shape[0]/4) : shape[0] - 25, 0:shape[1]]
    img = cv2.resize(img, (img_columns, img_rows), interpolation = cv2.INTER_AREA)
    return img

def process_new_image(name):
    """
    Import and normalize new images

    Parameters
    ----------
    name             : name of image file

    Returns
    -------
    An image
    """
    #preprocess the image
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.-.5
    return img

# small angle shift to account for the three cameras
CAMERA_SHIFT = 0.25

def preprocess_image_train(csv_line):
    """
    Imports each training image (left, right, and center) and shifts the left
    and right images to account for the offset from the center camera. Then,
    each augmentation function above is applied and the image is augmented.

    Parameters
    ----------
    csv_line             : .csv file containging the image locations and other
                           parameters such as the steering angle

    Returns
    -------
    The new image as an array and the corresponding steering angle.
    """
    k = np.random.randint(3) # one each for center, left, right

    if (k == 0):
        file = csv_line["left"][0].strip()
        view_shift = CAMERA_SHIFT
    elif(k == 1):
        file = csv_line["center"][0].strip()
        view_shift = 0
    elif(k == 2):
        file = csv_line["right"][0].strip()
        view_shift = -CAMERA_SHIFT

    steering_angle = csv_line["steering_angle"][0] + view_shift

    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, steering_angle, translate_x = height_width_shift(img, steering_angle, 100)
    img = brightness_shift(img)
    img = preprocessImage(img)
    img = np.array(img)

    # randomly flip the image
    if np.random.random() < 0.5:
        img, steering_angle = horizontal_flip(img, steering_angle)

    return img, steering_angle

def preprocess_image_predict(csv_line):
    path = csv_line['center'][0].strip()
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessImage(img)
    img = np.array(img)
    return img

def generate_batch_train(data, batch_size = 32):
    batch_images = np.zeros((batch_size, img_rows, img_columns, img_channels))
    steering_batch = np.zeros(batch_size)

    while 1:
        for batch_index in range(batch_size):
            index = np.random.randint(len(data))
            line_data = data.iloc[[index]].reset_index()

            keep = 0
            while keep == 0:
                x, y = preprocess_image_train(line_data)

                if abs(y) < 0.15:
                    ind = np.random.uniform()
                    if ind > threshold:
                        keep = 1
                else:
                    keep = 1

            batch_images[batch_index] = x
            steering_batch[batch_index] = y
        yield batch_images, steering_batch

def generate_batch_valid(data):
    while 1:
        for line_index in range(len(data)):
            line_data = data.iloc[[line_index]].reset_index()
            x = preprocess_image_predict(data)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = line_data['steering_angle'][0]
            y = np.array([[y]])
            yield x, y

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_rows, img_columns, img_channels), output_shape=(img_rows, img_columns, img_channels)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu', name='Conv1'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv2'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", activation='elu', name='Conv3'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, activation='elu', name='FC1'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, name='output'))
    return model

model = get_model()
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse', metrics = [])

class LifecycleCallback(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        global threshold
        threshold = 1 / (epoch + 1)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        print('BEGIN TRAINING')
        self.losses = []

    def on_train_end(self, logs={}):
        print('END TRAINING')

BATCH = 256
val = len(driving_data)

lifecycle_callback = LifecycleCallback()

validation_generator = generate_batch_valid(driving_data)
train_generator = generate_batch_train(driving_data, BATCH)

nb_vals = np.round(len(driving_data)/val) - 1
history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              samples_per_epoch = 20224, nb_epoch= 50,
                              nb_val_samples = val, verbose = 1,
                              callbacks = [lifecycle_callback])

model.save('model_test.h5')
