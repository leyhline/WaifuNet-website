# -*- coding: utf-8 -*-
"""
Return predicted probabilities for image classification.

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""


# Change backend of matplotlib so it needs no display server.
import matplotlib
matplotlib.use("Agg")

import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout


class SimpleConvNet:
    MAPPING = np.array(("Dress", "Nude", "School Uniform", "Swimsuit"),
                       dtype=np.unicode)
    TARGET_SIZE = 200
    FEATURE_DETECTOR = cv2.AKAZE_create()
    PLOT_COLORS = ('r', 'm', 'y', 'b')

    def __init__(self):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(200, 200, 3), activation="relu",
                                name="block1_conv1", init="glorot_normal"))
        model.add(MaxPooling2D(name="block1_pool"))
        model.add(Convolution2D(64, 3, 3, activation="relu", init="glorot_normal",
                                name="block2_conv1"))
        model.add(MaxPooling2D(name="block2_pool"))
        model.add(Convolution2D(128, 3, 3, activation="relu", init="glorot_normal",
                                name="block3_conv1"))
        model.add(MaxPooling2D(name="block3_pool"))
        model.add(Convolution2D(256, 3, 3, activation="relu", init="glorot_normal",
                                name="block4_conv1"))
        model.add(MaxPooling2D(name="block4_pool"))
        model.add(Flatten(name="flatten"))
        model.add(Dense(256, activation="relu", name="fc1", init="glorot_normal"))
        model.add(Dropout(0.50, name="dropout1"))
        model.add(Dense(4, activation="softmax", init="glorot_normal",
                        name="predictions"))
        model.load_weights(join(dirname(__file__), "weights.hdf5"))
        self.model = model

    def crop(self, img):
        """
        Crop image to make it quadratic using a feature detector for finding a nice center.
        (I know, feature detectors are not for something like this...)
        """
        kp = self.FEATURE_DETECTOR.detect(img)
        ysize, xsize = img.shape[:2]
        smallest = min(xsize, ysize)
        if not kp:
            print("No keypoints. Crop image in the center.")
            if xsize > ysize:
                dst = img[:, xsize // 2 - smallest // 2:xsize // 2 + smallest // 2]
            else:
                dst = img[xsize // 2 - smallest // 2:xsize // 2 + smallest // 2,:]
            return dst
        kp_sum = np.zeros(abs(xsize - ysize))
        if xsize > ysize:
            x_or_y = 0
        else:
            x_or_y = 1
        for i in range(kp_sum.size):
            for k in kp:
                if k.pt[x_or_y] - i < smallest:
                    kp_sum[i] += k.response
        imax = kp_sum.argmax()
        if x_or_y:
            dst = img[imax:imax+smallest,:]
        else:
            dst = img[:,imax:imax+smallest]
        return dst

    def preprocess(self, img):
        """Preprocesses given image. Using crop and resize if necessary.
        Also adds another dimension."""
        ysize, xsize = img.shape[:2]
        # Enlarge image if it is smaller than 200x200.
        if xsize < self.TARGET_SIZE:
            img = cv2.resize(img, (self.TARGET_SIZE, self.TARGET_SIZE / xsize * ysize))
            ysize, xsize = img.shape[:2]
        if ysize < self.TARGET_SIZE:
            img = cv2.resize(img, (self.TARGET_SIZE / ysize * xsize, self.TARGET_SIZE))
            ysize, xsize = img.shape[:2]
        # Resize image.
        if xsize > ysize:
            xsize = round(xsize / ysize * self.TARGET_SIZE)
            ysize = self.TARGET_SIZE
        elif xsize < ysize:
            ysize = round(ysize / xsize * self.TARGET_SIZE)
            xsize = self.TARGET_SIZE
        else:
            xsize = ysize = self.TARGET_SIZE
        dst = cv2.resize(img, (xsize, ysize), interpolation=cv2.INTER_AREA)
        # Crop image if it is not already quadratic.
        if not xsize == ysize:
            dst = self.crop(dst)
        return dst

    def predict(self, img):
        """Use neural network to predict class of image. Needs an image of size 200x200."""
        img = np.float32(img) / 255
        img = np.expand_dims(img, axis=0)
        return self.model.predict_on_batch(img)

    def plot_prediction(self, img):
        """Returns an IO object with a plot of the predictions and the image."""
        img = self.preprocess(img)
        data = self.predict(img)[0]
        f, (plot, image) = plt.subplots(1, 2)
        bars = plot.barh(np.arange(len(data)), data, height=1, align="center",
                                  tick_label=('Dress', 'Nude', 'School', 'Swimsuit'))
        for i in range(len(bars)):
            bars[i].set_color(self.PLOT_COLORS[i])
        # BGR -> RGB
        img = img[...,::-1]
        image.axis("off")
        image.imshow(img)
        # Read plot into BytesIO project which can be handled like a file.
        file = io.BytesIO()
        plt.savefig(file)
        file.seek(0)
        return file