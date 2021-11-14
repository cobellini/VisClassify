import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow as tf

from flask import Flask, request, redirect, url_for, render_template
from tensorflow import keras
from keras.applications.resnet import ResNet50
from numpy.core.fromnumeric import resize
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.backend import GraphExecutionFunction, set_session
from PIL import Image
from tensorflow.python.keras.regularizers import L2
from Classifier import Classifier
from DatasetGenerator import DatasetGenerator

#run this to create the web app.
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    my_image_re = resize(my_image, (300,300,1))   
    reshaped_images = my_image_re.astype('float32')

    normalised_images = reshaped_images/255.0
    class_labels = ['area_chart', 'bar_chart', 'box_plot', 'chord_diagram', 'donut_chart', 'error_bar', 
    'flow_diagram', 'glyph_based', 'graph', 'heatmap', 'hierarchical_edge_bundling', 'line_chart', 'map', 
    'matrix', 'parallel_coordinate', 'pie_chart', 'polar_plot', 'proportional_area_chart', 'sankey_diagram', 
    'scatterplot', 'sector_chart', 'small_multiple', 'storyline', 'stripe_graph', 'sunburst_icicle', 'table', 
    'tree', 'treemap', 'unit_visualization', 'word_cloud']
    graph_obj = tf.Graph()
    with graph_obj.as_default():
        model = load_model()
        label_prediction = model.predict(np.array([reshaped_images,]))[0,:]
        
    predictions = {"visType" : class_labels[np.argmax(label_prediction)]}
    return render_template('prediction.html', predictions=predictions)

def load_model():
    classifier_obj = Classifier(300, None, None)
    model = classifier_obj.new_model()
    model.build(input_shape=(None, 300, 300, 1))
    model.load_weights('Basic_CNN')
    optimizer_learning_rate = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer_learning_rate, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

app.run(host='0.0.0.0', port=50000)