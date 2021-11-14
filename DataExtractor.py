import json

from PIL import Image

class DataExtractor:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_data_to_json(self):
        file = open(self.filepath)
        file_data = json.load(file)
        return file_data

    def save_data(self, jsonData, save_location):
        jsonFile = json.dumps(jsonData)
        jsonVisFile = open(save_location, 'w')
        jsonVisFile.write(jsonFile)
        jsonVisFile.close()
    
    def extract_visualisations(self, data, imageslocation, image_save_location):
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
                image = Image.open(imageslocation + key + '/' + str(file_no) + '.' + file_data[1])           
                for visType in file_img_bbx:    
                    bbox_coords = file_img_bbx[visType]
                    for y in range(len(bbox_coords)):
                        json_obj_data = {}
                        imageNumber = str(img_count)        
                        img = image.crop((bbox_coords[y][0], bbox_coords[y][1],bbox_coords[y][2],bbox_coords[y][3]))
                        img.save(image_save_location % imageNumber)
                        json_obj_data['Image_path'] = (imageNumber)
                        json_obj_data['Vis_type'] = visType
                        jsonData.append(json_obj_data)
                        img_count = img_count + 1
        return jsonData                