import os
from keras.applications.resnet import ResNet50
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from Classifier import Classifier
from DataExtractor import DataExtractor
from DatasetGenerator import DatasetGenerator
from ClassifierEvaluation import ClassifierEvaluation
from numpy.core.fromnumeric import resize
from sklearn.utils import class_weight

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
PIXELS = 100
EPOCHS = 30

#run this to extract files, create dataset and build/train model. 

#Change filepaths to reflect your directory structure.
VISIMAGES_JSON_PATH = '../data/data/vis_data/visimages_data.json'
VISIMAGES_IMAGES_PATH = '../data/data/vis_data/images/'
EXTRACTED_IMAGES_SAVE_PATH = '../data/data/vis_data/images_corrected/images/%s.png'
EXTRACTED_IMAGES_JSON__SAVE_PATH = '../data/data/vis_data/images_corrected/VisImages.json'
EXTRACTED_IMAGES_DIRECTORY_PATH = '../data/data/vis_data/images_corrected/images/'
MODEL_WEIGHTS_SAVE_NAME_PATH = 'dissertation_model_v2'
   
def extract_images():
    extraction_obj = DataExtractor(VISIMAGES_JSON_PATH)
    dataset_json = extraction_obj.read_data_to_json()
    extracted_visualisations_json = extraction_obj.extract_visualisations(dataset_json, VISIMAGES_IMAGES_PATH, EXTRACTED_IMAGES_SAVE_PATH)
    extraction_obj.save_data(extracted_visualisations_json, EXTRACTED_IMAGES_JSON__SAVE_PATH)

def generate_data():
    generation_obj = DatasetGenerator(PIXELS, EXTRACTED_IMAGES_DIRECTORY_PATH, EXTRACTED_IMAGES_JSON__SAVE_PATH)
    extracted_json_info = generation_obj.read_data_to_json()
    class_labels = generation_obj.generate_class_labels(extracted_json_info)
    dataset = generation_obj.generate_workable_dataset(extracted_json_info, class_labels)
    return class_labels, dataset

def train_dataset(dataset, class_labels):
    classifier_obj = Classifier(PIXELS, class_labels, dataset)  
    train_image, train_labels, validation_image, validation_label, prediction_image, prediction_label = classifier_obj.data_preprocessing()
    model = classifier_obj.construct_model()
    class_weights = {0: 29.7, 
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
    optimizer_learning_rate = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer_learning_rate, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_history =  model.fit(train_image, train_labels, validation_data=(validation_image, validation_label), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=ReduceLROnPlateau(), class_weight=class_weights)
    model.save_weights(MODEL_WEIGHTS_SAVE_NAME_PATH)
    return model_history

def generate_metrics(model_history):
    metrics_obj = ClassifierEvaluation(model_history)
    metrics_obj.classifier_accuracy()
    metrics_obj.classifier_error()

def test_new(dataset, class_labels):

    classifier_obj = Classifier(PIXELS, class_labels, dataset)  
    train_image, train_labels, validation_image, validation_label = classifier_obj.data_preprocessing()
    
    class_weights = {0: 5, 1: 1, 2: 18.8, 3: 40.1, 4: 13.7, 5: 7.2, 6: 5, 7: 9.7, 8: 1.2,
     9: 1.60, 10: 27.7, 11: 1.70, 12: 2.44, 13: 3.18, 14: 5.24, 15: 16.8, 16: 41.4, 17: 17.8,
      18: 19.6, 19: 1.2, 20: 92.7, 21: 7.4, 22: 110.8, 23: 21.3, 24: 19.6, 25: 2.35, 26: 3.95,
       27: 9.4, 28: 18.6, 29: 13.0}
    
    model = keras.Sequential([  
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        keras.layers.experimental.preprocessing.RandomRotation(0.8),
    ])
    wrapper_model = ResNet50(include_top=False, input_shape=(PIXELS, PIXELS, 3), classes=30, weights='imagenet')
    for layer in wrapper_model.layers:
        layer.trainable=False

    model.add(wrapper_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(250, activation='relu'))
    model.add(keras.layers.Dense(30, activation='softmax'))

    optimizer_learning_rate = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # model.compile(optimizer=optimizer_learning_rate, loss='categorical_crossentropy', metrics=['accuracy'])
    # model_history =  model.fit(train_image, train_labels, validation_data=(validation_image, validation_label), epochs=100, batch_size=BATCH_SIZE, callbacks=ReduceLROnPlateau(), class_weight=class_weights)

    # model.save_weights('training_checkpoint_v6')

    model.load_weights('checkpoint')

    for layer in wrapper_model.layers:
        layer.trainable = True
   
    model.compile(optimizer=optimizer_learning_rate, loss='categorical_crossentropy', metrics=['accuracy'])
    model_history =  model.fit(train_image, train_labels, validation_data=(validation_image, validation_label), epochs=100, batch_size=BATCH_SIZE, callbacks=ReduceLROnPlateau(), class_weight=class_weights)

    model.save_weights(MODEL_WEIGHTS_SAVE_NAME_PATH)

    return model_history

def test_prediction():
    class_labels, dataset = generate_data()
    classifier_obj = Classifier(PIXELS, class_labels, dataset)  
    train_image, train_labels, validation_image, validation_label, prediction_image, prediction_label = classifier_obj.data_preprocessing()
    model = classifier_obj.construct_model()
    model.build(input_shape=(None, 100, 100, 3))
    model.load_weights('dissertation_model')
    optimizer_learning_rate = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer_learning_rate, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    prediction = model.predict(prediction_image)
    
    for x in range(len(prediction)):
        print(prediction[x])
   

def run():
    print('RUNNING....')
    #must be run the first time if you need to create a valid directory and json file for your dataset..
    #extract_images()

    #create workable dataset and class labels for dataset..
    print('CREATING DATASET...')
    class_labels, dataset = generate_data()
    #get classifier accuracy and save weights..
    print('TRAINING MODEL...')
    classifier_history = train_dataset(dataset, class_labels)
    generate_metrics(classifier_history)
    return None
   
run()
#test_prediction()









