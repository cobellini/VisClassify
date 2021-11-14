
import numpy
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet import ResNet50
from tensorflow.python.keras.regularizers import L2
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from PIL import Image


class Classifier:
    def __init__(self, pixels, class_label, dataset):
        self.pixels = pixels
        self.class_label = class_label
        self.dataset = dataset

    def data_preprocessing(self):   
        dataset_numpy = numpy.array(self.dataset)
        numpy.random.shuffle(dataset_numpy)
        shuffled_images = dataset_numpy[:,:-1]
        shuffled_labels = dataset_numpy[:,-1:]    
        reshaped_images = numpy.reshape(shuffled_images, (len(shuffled_images), self.pixels, self.pixels, 3))
        reshaped_images = reshaped_images.astype('float32')      
        train_images = reshaped_images[:-7000]
        train_labels = shuffled_labels[:-7000]          
        test_images = reshaped_images[len(reshaped_images)-6000:]
        test_labels = shuffled_labels[len(shuffled_labels)-6000:] 
        prediction_images = reshaped_images[len(reshaped_images)-1000:]
        prediction_labels = shuffled_labels[len(shuffled_labels)-1000:] 
        return train_images, train_labels, test_images, test_labels, prediction_images, prediction_labels

    def construct_model(self):
        model = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            keras.layers.experimental.preprocessing.RandomRotation(0.8),
        ])
        wrapper_model = ResNet50(include_top=False, input_shape=(self.pixels, self.pixels, 3), pooling='avg',
        classes=30, weights='imagenet')
        for layer in wrapper_model.layers[:-250]:
            layer.trainable=False

        model.add(wrapper_model)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(30, activation='softmax'))
        return model


    def new_model(self):
        model = keras.Sequential([
            keras.layers.AveragePooling2D(6, 3, input_shape=(self.pixels, self.pixels, 1)),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(30, activation='softmax')
        ])
        return model