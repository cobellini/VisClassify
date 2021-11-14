import json
from PIL import Image
import numpy

class DatasetGenerator:
    def __init__(self, pixels, images_path, images_info_path):
        self.pixels = pixels
        self.images_path = images_path
        self.images_info_path = images_info_path

    def read_data_to_json(self):
        file = open(self.images_info_path)
        file_data = json.load(file)
        return file_data

    def generate_class_labels(self, json_data):
        vis_list = []
        class_labels = {}
        for x in range(len(json_data)):
            vis_list.append(json_data[x]['Vis_type'])
            np_lov = numpy.array(vis_list)  
            unique_vis = numpy.unique(np_lov)

        for z in range(len(unique_vis)):
            class_labels[unique_vis[z]] = z 
        return class_labels

    def generate_workable_dataset(self, json_data, class_labels):
        workable_data = []
        for z in range(len(json_data)):
            image = numpy.array(Image.open(self.images_path + str(z) +'.png').convert('RGB').resize((self.pixels, self.pixels)))     
            vis_image_and_label = numpy.append(image, class_labels[json_data[z]['Vis_type']])
            workable_data.append(vis_image_and_label)
        
        return workable_data

    