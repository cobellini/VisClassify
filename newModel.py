import os
from keras.applications.resnet import ResNet50
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers.core import Activation
from Classifier import Classifier
from DataExtractor import DataExtractor
from DatasetGenerator import DatasetGenerator
from ClassifierEvaluation import ClassifierEvaluation
from numpy.core.fromnumeric import resize
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from collections import Counter
#Change filepaths to reflect your directory structure.
VISIMAGES_JSON_PATH = '../data/data/vis_data/visimages_data.json'
VISIMAGES_IMAGES_PATH = '../data/data/vis_data/images/'
EXTRACTED_IMAGES_SAVE_PATH = '../data/data/vis_data/images_corrected/images/%s.png'
EXTRACTED_IMAGES_JSON__SAVE_PATH = '../data/data/vis_data/images_corrected/VisImages.json'
EXTRACTED_IMAGES_DIRECTORY_PATH = '../data/data/vis_data/images_corrected/images/'
MODEL_WEIGHTS_SAVE_NAME_PATH = 'Basic_CNN_Update'

BATCH_SIZE = 32
LEARNING_RATE = 0.001
PIXELS = 100
EPOCHS = 5

def get_Dataset():
    generation_obj = DatasetGenerator(PIXELS, EXTRACTED_IMAGES_DIRECTORY_PATH, EXTRACTED_IMAGES_JSON__SAVE_PATH)
    extracted_json_info = generation_obj.read_data_to_json()
    class_labels = generation_obj.generate_class_labels(extracted_json_info)
    dataset = generation_obj.generate_workable_dataset(extracted_json_info, class_labels)
    return class_labels, dataset


def test_new(dataset, class_labels):
    
    classifier_obj = Classifier(PIXELS, class_labels, dataset)  
    train_image, train_labels, validation_image, validation_label = classifier_obj.data_preprocessing()

    # pleb = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)

    class_weights = {
    0: 29.7, 
    1: 6.7, 
    2: 120, 
    3: 252.3, 
    4: 88.2, 
    5: 46.07, 
    6: 30.4, 
    7: 63.0, 
    8: 9.11,
    9: 10.5, 
    10: 171.9, 
    11: 11.1, 
    12: 15.9, 
    13: 21.5, 
    14: 33.9, 
    15: 116.8, 
    16: 296.7, 
    17: 115.8,
    18: 136.3, 
    19: 8.05, 
    20: 642.8, 
    21: 48.3, 
    22: 692.3, 
    23: 134.3,
    24: 133.0, 
    25: 15.6, 
    26: 26.7,
    27: 63.8, 
    28: 127.9, 
    29: 85.1}
    
    model = keras.Sequential([
        keras.layers.AveragePooling2D(6, 3, input_shape=(PIXELS, PIXELS, 1)),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(30, activation='softmax')
    ])
    
    optimizer_learning_rate = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer_learning_rate, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_history =  model.fit(train_image, train_labels, validation_data=(validation_image, validation_label), epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weights)

    model.save_weights(MODEL_WEIGHTS_SAVE_NAME_PATH)

    return model_history




class_labels, dataset = get_Dataset()

history = test_new(dataset, class_labels)

metrics_obj = ClassifierEvaluation(history)
metrics_obj.classifier_accuracy()
metrics_obj.classifier_error()

