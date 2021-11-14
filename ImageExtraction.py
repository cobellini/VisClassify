import math
from matplotlib import pyplot as plt
import numpy
import json
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers.core import Activation
from keras.applications.resnet import ResNet50
from tensorflow.python.keras.regularizers import L2

def extract_visualisations():
    file = open('../data/data/vis_data/visimages_data.json')
    data = json.load(file)
    jsonData = []
    img_count = 0
    for key in data:
        img_info = {}
        
        img_file = data[key]
        for x in range(len(img_file)):
            
            file_img_bbx = img_file[x]['visualization_bbox']
            vis_numbers = img_file[x]['nums_of_visualizations']
            if(len(vis_numbers) == 0):
                continue
            img_info['image_id'] = img_file[x]['image_id']

            img_info['image_file_name'] = img_file[x]['file_name']
            file_data = img_info['image_file_name'].split('.')
            file_key_number = file_data[0].split('_')
            file_no = int(file_key_number[1])
            image = Image.open('../data/data/vis_data/images/' + key + '/' + str(file_no) + '.' + file_data[1])
            
            for visType in file_img_bbx:    
                bbox_coords = file_img_bbx[visType]
                for y in range(len(bbox_coords)):
                    json_obj_data = {}
                    imageNumber = str(img_count)        
                    img = image.crop((bbox_coords[y][0], bbox_coords[y][1],bbox_coords[y][2],bbox_coords[y][3]))
                    img.save('../data/data/vis_data/images_corrected/images/%s.png' % imageNumber)
                    json_obj_data['Image_path'] = (imageNumber)
                    json_obj_data['Vis_type'] = visType
                    jsonData.append(json_obj_data)
                    img_count = img_count + 1
    jsonFile = json.dumps(jsonData)
    jsonVisFile = open('../data/data/vis_data/images_corrected/VisImages.json', 'w')
    jsonVisFile.write(jsonFile)
    jsonVisFile.close()

def generate_class_labels(json_data):
  List_of_Vis = []
  for x in range(len(json_data)):
    List_of_Vis.append(json_data[x]['Vis_type'])
    np_lov = numpy.array(List_of_Vis)
    
    unique_vis = numpy.unique(np_lov)

  class_labels = {}
  
  for z in range(len(unique_vis)):
    class_labels[unique_vis[z]] = z
    # class_labels[unique_vis[z]] = z + 1

  print(class_labels)
 
  return class_labels

def generate_data(pixels):    
    file = open('../data/data/vis_data/images_corrected/VisImages.json')
    json_data = json.load(file)
    class_labels= generate_class_labels(json_data)
    
    workable_data = []
    
    for z in range(len(json_data)):
        image = numpy.array(Image.open('../data/data/vis_data/images_corrected/images/' + str(z) +'.png').convert('RGB').resize((pixels, pixels)))
        
        vis_image_and_label = numpy.append(image, class_labels[json_data[z]['Vis_type']])
        workable_data.append(vis_image_and_label)
        
    return workable_data

def data_preprocessing(data, pixels, mu=0.15):
    training = numpy.array(data)
    numpy.random.shuffle(training)
    shuffled_images = training[:,:-1]
    shuffled_labels = training[:,-1:]
    
    #reshaped_shuffled_images = numpy.reshape(shuffled_images, (len(shuffled_images), pixels, pixels, 3))
    reshaped_shuffled_images = numpy.reshape(shuffled_images, (len(shuffled_images), pixels, pixels, 3))
    #grayscaled_images = reshaped_shuffled_images/255.0

    train_image = reshaped_shuffled_images[:-6000]
    train_labels = shuffled_labels[:-6000]
    test_images = reshaped_shuffled_images[len(reshaped_shuffled_images)-6000:]
    test_labels = shuffled_labels[len(reshaped_shuffled_images)-6000:]

    unique, counts = numpy.unique(train_labels, return_counts=True)
    label_weights = dict(zip(unique, counts))

    total = numpy.sum(list(label_weights.values()))
    keys = label_weights.keys()
    class_weight = dict()

    for key in keys:
      score = math.log(mu*total/float(label_weights[key]))
      class_weight[key] = score if score > 1.0 else 1.0
   
    return train_image, train_labels, test_images, test_labels, class_weight

def classifier_v1(dataset, pixels):
  
  train_image, train_labels, test_images, test_labels, class_weights = data_preprocessing(dataset, pixels)
  
  model = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    keras.layers.experimental.preprocessing.RandomRotation(0.8),
  ])
  model.add(keras.layers.Conv2D(64, (3,3), input_shape = (pixels, pixels, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(64))
  model.add(keras.layers.Activation('relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(30))
  model.add(keras.layers.Activation("softmax"))

  optimizer_learning_rate = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer_learning_rate, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
  history = model.fit(train_image, train_labels, validation_data=(test_images, test_labels), epochs=30, batch_size=32, class_weight=class_weights ,callbacks=ReduceLROnPlateau())
  model.summary()
  return history

def resnet50_model_adaption(dataset, pixels):
  train_image, train_labels, test_images, test_labels, class_weights = data_preprocessing(dataset, pixels)

  model = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    keras.layers.experimental.preprocessing.RandomRotation(0.8),
  ])
  pre_model = ResNet50(include_top=False, input_shape=(pixels, pixels, 3), pooling='avg', classes=30, weights="imagenet")

  for layer in pre_model.layers[:-200]:
        layer.trainable=False

  model.add(pre_model)
  model.add(keras.layers.Dense(64, kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01)))
  model.add(keras.layers.Activation("relu"))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(30))
  model.add(keras.layers.Activation("softmax"))

  optimizer_learning_rate = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer_learning_rate, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.build(input_shape=(None, pixels, pixels, 3))
  history = model.fit(train_image, train_labels, validation_data=(test_images, test_labels), epochs=30, batch_size=64, callbacks=ReduceLROnPlateau())
  model.summary()
  #model.save_weights('cnn_model_weights.h5')
  return history














