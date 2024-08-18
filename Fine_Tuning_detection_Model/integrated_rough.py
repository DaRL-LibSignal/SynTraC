import torch
import cv2
import numpy as np
import os
import glob as glob
import json

from xml.etree import ElementTree as et
from config import (
	DEVICE, CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn, ssdlite320_mobilenet_v3_large
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from PIL import Image
import numpy as np

# from ....PythonAPI.examples.rough.RL_Model.Infinity_traffic_time.object_detection_model import Object_Detection_Model
# import sys
# sys.path.append('/home/local/ASURITE/hmei7/Documents/PythonAPI/examples/rough/RL_Model/Infinity_traffic_time')

# from object_detection_model import Object_Detection_Model




#Modeify image path
def processed_image_path(source_path, folder):
	destination_directory = '/home/local/ASURITE/tchen169/Documents/CV4TSC/rough/RL_Model/cyclic_traffic_time_for_train/1fpsDataset'


	# Extract the filename
	filename = os.path.basename(source_path)

	# Construct the new path
	new_path = os.path.join(destination_directory, folder, filename)

	# print("New path:", new_path)
	return new_path

class Object_Detection_Model:
	models = {
			"Faster R-CNN": fasterrcnn_resnet50_fpn(pretrained=True,weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
			"Mask R-CNN": maskrcnn_resnet50_fpn(pretrained=True),
			"RetinaNet": retinanet_resnet50_fpn(pretrained=True),
			#"SSD": ssdlite320_mobilenet_v3_large(pretrained=True),  # SSD variant from torchvision
		}
	def __init__(self, model_name) -> None:
		self.model = self.load_model(model_name)

	def load_model(self, model_name):
		model = Object_Detection_Model.models[model_name]
		# Set the model to evaluation mode
		model.eval()
		# Move the model to the specified device
		model.to(DEVICE)
		return model
	
	def get_only_predictions(self, image_path, image):
		if image_path != None:
			# Load and transform the image
			image = Image.open(image_path).convert("RGB")
			transform = transforms.Compose([transforms.ToTensor()])
			image_tensor = transform(image).to(DEVICE)
		else:
   
			# print("heyyyo", type(image))
			# Apply the transformation to your PIL image
   			
			image_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).to(DEVICE)


		# Get predictions from the model
		with torch.no_grad():
			prediction = self.model([image_tensor])
			
		return prediction


class tp(Dataset):
	def __init__(self, isTrain , dir_path, width, height, classes, transforms=None):
		self.transforms = transforms
		self.dir_path = dir_path
		self.height = height
		self.width = width
		self.classes = classes
		self.image_file_types = ['*.jpg']
		self.all_image_paths = []
		self.isTrain = isTrain
		self.train_split = 0.4
		
		# Get all the image paths in sorted order.
		self.all_image_paths = self.load_and_process_image_path(self.dir_path)
		self.all_images = [(image_path.split(os.path.sep)[-2]+image_path.split(os.path.sep)[-1]) for image_path in self.all_image_paths]
		self.object_detection_model = Object_Detection_Model("RetinaNet")
		# self.all_images = sorted(self.all_images)
			
	def load_and_process_image_path(self, dir_path):
		all_image_paths = []
		
		with open(dir_path, 'r') as file:
			camera_json = json.load(file)
			n_limit = int(len(camera_json)*(self.train_split))
			# print(f"n_limit {n_limit}")
			if(self.isTrain):
				for n in range(0,n_limit):
					camera_data = camera_json[str(n)]
					for i in range(4):
						folder= chr(ord('A') + i)
						# print(f"{n} {folder}")
						image_path = camera_data[folder]["image"]
						image_path = processed_image_path(image_path, folder)
						all_image_paths.append(image_path)
			elif(self.isTrain==False):
				# for n in range(n_limit, len(camera_json)):
				for n in range(n_limit, n_limit+100):
					camera_data = camera_json[str(n)]
					for i in range(4):
						folder= chr(ord('A') + i)
						# print(f"{n} {folder}")
						image_path = camera_data[folder]["image"]
						image_path = processed_image_path(image_path, folder)
						all_image_paths.append(image_path)
		# print(f"all_image_paths {len(all_image_paths)}")
		return all_image_paths
						
	def __getitem__(self, idx):
		# Capture the image name and the full image path.
		full_name = self.all_images[idx]
		folder_name = full_name[0]
		image_name = full_name[1:] #image_name = 100.jpg
		image_path = self.all_image_paths[idx]
		

		# Read and preprocess the image.
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
		image_resized = cv2.resize(image, (self.width, self.height))
		image_resized /= 255.0
		
		# Capture the corresponding XML file for getting the annotations.
		# annot_filename = os.path.splitext(image_name)[0] + '.xml'
		# annot_file_path = os.path.join(self.dir_path, annot_filename)
		
		boxes = []
		labels = []
		# tree = et.parse(annot_file_path)
		# root = tree.getroot()
		
		# Original image width and height.
		image_width = image.shape[1]
		image_height = image.shape[0]
		
		# Box coordinates for xml files are extracted 
		# and corrected for image size given.
		with open(self.dir_path, 'r') as file:
			camera_json = json.load(file)
			n = os.path.splitext(image_name)[0]
			camera_data = camera_json[str(n)][folder_name]
			# label = [camera_data[folder]["Lane1"]["total_vehicles"], camera_data[folder]["Lane2"]["total_vehicles"]]
			Lanes = ["Lane1", "Lane2"]
			for lane in Lanes:
				for _,member in enumerate(camera_data[lane]["vehicles"]):
					labels.append(1)
					
					xmin, ymin, xmax, ymax = member["coordinates"]
					
					# Resize the bounding boxes according 
					# to resized image `width`, `height`.
					xmin_final = (xmin/image_width)*self.width
					xmax_final = (xmax/image_width)*self.width
					ymin_final = (ymin/image_height)*self.height
					ymax_final = (ymax/image_height)*self.height

					# Check that max coordinates are at least one pixel
					# larger than min coordinates.
					if xmax_final == xmin_final:
						xmax_final += 1
					if ymax_final == ymin_final:
						ymax_final += 1
					# Check that all coordinates are within the image.
					if xmax_final > self.width:
						xmax_final = self.width
					if ymax_final > self.height:
						ymax_final = self.height
					
					boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
					

		# From Object detection model get bounding boxes.
		predictions = self.object_detection_model.get_only_predictions(None, image_resized)
		vehicle_labels = [2, 3, 4, 6, 8]  
		# Convert vehicle_labels list to a tensor and move it to the same self.device as the model's predictions
		vehicle_labels_tensor = torch.tensor(vehicle_labels).to(predictions[0]['labels'].device)

		if len(predictions) > 0:
			car_indices = torch.isin(predictions[0]['labels'], vehicle_labels_tensor) & (predictions[0]['scores'] > 0.5)
			car_boxes = predictions[0]['boxes'][car_indices]
			car_scores = predictions[0]['scores'][car_indices]
			car_labels = predictions[0]['labels'][car_indices]
			
		boxes = torch.FloatTensor(boxes).to(DEVICE)
		boxes = boxes.view(-1,4)
		# print(f"Total Detections by detection model for camera: {car_boxes.size()}")
		# print("Ground Truth byy carla", (boxes.size()))

		# Merge car_boxes with boxes
		merged_boxes = torch.cat((car_boxes, boxes), dim=0)
		merged_scores = torch.cat((car_scores, torch.ones(len(boxes)).to(DEVICE)), dim=0)
		# print(f"Merged Boxes: {len(merged_boxes)}")
		# Perform NMS
  		
		iou_threshold = 0.5
		filtered_prediction = torchvision.ops.nms(merged_boxes, merged_scores, iou_threshold)
		boxes = merged_boxes[filtered_prediction]
		labels = torch.ones(len(filtered_prediction))
		# print(f"filtered_predictions {filtered_prediction}")

		#Apply NMS to remove duplicate boxes.

		
		# Bounding box to tensor.
		
		# boxes = torch.as_tensor(boxes, dtype=torch.float32)
		# # Area of the bounding boxes.
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
			else torch.as_tensor(boxes, dtype=torch.float32)
		# No crowd instances.
		iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
		# Labels to tensor.
		labels = torch.as_tensor(labels, dtype=torch.int64)

		# Prepare the final `target` dictionary.
		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["area"] = area
		target["iscrowd"] = iscrowd
		image_id = torch.tensor([idx])
		target["image_id"] = image_id
		
		# if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
		# 	target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
		return image_resized, target

	def __len__(self):
		return len(self.all_images)


# Prepare the final datasets and data loaders.
def create_train_dataset(DIR):
	isTrain = True
	train_dataset = tp(
		isTrain, DIR+'/Intersection_camera.json', RESIZE_TO, RESIZE_TO, CLASSES
	)
	return train_dataset

def create_valid_dataset(DIR):
	isTrain = False
	valid_dataset = tp(
		isTrain, DIR+'/Intersection_camera.json', RESIZE_TO, RESIZE_TO, CLASSES
	)
	return valid_dataset
def create_train_loader(train_dataset, num_workers=0):
	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=collate_fn,
		drop_last=True
	)
	return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
	valid_loader = DataLoader(
		valid_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=collate_fn,
		drop_last=True
	)
	return valid_loader


# execute `datasets.py`` using Python command from 
# Terminal to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
	# print("HI")
	# sanity check of the Dataset pipeline with sample visualization
	dataset = tp(
		True, TRAIN_DIR+'/Intersection_camera.json', RESIZE_TO, RESIZE_TO, CLASSES
	)
	print(f"Number of training images: {len(dataset)}")
	# print(dataset.all_images[:10])
	
	# function to visualize a single sample
	def visualize_sample(image, target):
		for box_num in range(len(target['boxes'])):
			box = target['boxes'][box_num]
			label = CLASSES[target['labels'][box_num]]
			# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			cv2.rectangle(
				image, 
				(int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
				(0, 0, 255), 
				2
			)
			cv2.putText(
				image, 
				label, 
				(int(box[0]), int(box[1]-5)), 
				cv2.FONT_HERSHEY_SIMPLEX, 
				0.7, 
				(0, 0, 255), 
				2
			)
		cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
		cv2.waitKey(0)
		
	NUM_SAMPLES_TO_VISUALIZE = 10
	for i in range(2000,2000+NUM_SAMPLES_TO_VISUALIZE):
		image, target = dataset[i]
		visualize_sample(image, target)