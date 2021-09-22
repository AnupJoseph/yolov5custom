# Standrard Exports
import os
import random

# External Exports
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# Project specific exports
from xml_parse import extract_info_from_xml
from convert_to_yolo import class_name_to_id_mapping, convert_to_yolov5
from plot import plot_bounding_box
from move import move_files_to_folder

#  Get the annotations
annotations = [
    os.path.join("/content/voc_plate_dataset/Annotations/", x)
    for x in os.listdir("/content/voc_plate_dataset/Annotations/")
    if x[-3:] == "xml"
]
annotations.sort()

# Convert and save the annotations
for ann in tqdm(annotations):
    info_dict = extract_info_from_xml(ann)
    convert_to_yolov5(info_dict)
annotations = [
    os.path.join("Annotations", x) for x in os.listdir("Annotations") if x[-3:] == "txt"
]

random.seed(42)

class_id_to_name_mapping = dict(
    zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys())
)

# Get any random annotation file
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x] for x in annotation_list]

# Get the corresponding image file
image_file = annotation_file.replace(
    "Annotations", "voc_plate_dataset/JPEGImages"
).replace("txt", "jpg")
assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)

# Plot the Bounding Box
plot_bounding_box(image, annotation_list)

# Read images and annotations
images = [
    os.path.join("/content/voc_plate_dataset/JPEGImages", x)
    for x in os.listdir("/content/voc_plate_dataset/JPEGImages/")
]
annotations = [
    os.path.join("/content/Annotations", x)
    for x in os.listdir("/content/Annotations")
    if x[-3:] == "txt"
]
images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')