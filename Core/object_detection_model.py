import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn, ssdlite320_mobilenet_v3_large
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from PIL import Image
import numpy as np
import json


class Object_Detection_Model:
	models = {
			"Faster R-CNN": fasterrcnn_resnet50_fpn(pretrained=True,weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
			"Mask R-CNN": maskrcnn_resnet50_fpn(pretrained=True),
			"RetinaNet": retinanet_resnet50_fpn(pretrained=True),
			#"SSD": ssdlite320_mobilenet_v3_large(pretrained=True),  # SSD variant from torchvision
		}
	def __init__(self, model_name) -> None:
		# Check if CUDA is available
		if torch.cuda.is_available():
			self.device = torch.device("cuda")
			print("CUDA is available. Model will be moved to GPU.")
		else:
			self.device = torch.device("cpu")
			print("CUDA is not available. Model will run on CPU.")
		self.model = self.load_model(model_name)

	def load_model(self, model_name):
		model = Object_Detection_Model.models[model_name]
		# Set the model to evaluation mode
		model.eval()
		# Move the model to the specified device
		model.to(self.device)
		return model
	
	def point_in_search_region(cx, cy, ax, ay, bx, by):
			# Compute the vectors C->B and A->B
			vector_CB = (bx - cx, by - cy)
			vector_AB = (bx - ax, by - ay)

			# Compute the dot product of vectors C->B and A->B
			dot_product = vector_CB[0] * vector_AB[0] + vector_CB[1] * vector_AB[1]
			if dot_product>0:
				return True
			return False
	
	def process_image_for_detections(self, image_path, image, lane_2d_data, folder, distance_threshold):
		if image_path != None:
			# Load and transform the image
			image = Image.open(image_path).convert("RGB")
			transform = transforms.Compose([transforms.ToTensor()])
			image_tensor = transform(image).to(self.device)
		else:
			# Convert BGRA to RGB
			img_rgb = image[:, :, :3][:, :, ::-1]

			# Convert the NumPy array to a PIL Image
			image_pil = Image.fromarray(img_rgb)

			# Define the transformation
			transform = transforms.Compose([
				transforms.ToTensor(),
			])

			# Apply the transformation to your PIL image
			image_tensor = transform(image_pil).to(self.device)

		# Get predictions from the model
		with torch.no_grad():
			prediction = self.model([image_tensor])

		vehicle_labels = [2, 3, 4, 6, 8]  
		# Convert vehicle_labels list to a tensor and move it to the same self.device as the model's predictions
		vehicle_labels_tensor = torch.tensor(vehicle_labels).to(prediction[0]['labels'].device)

		print(f"Total Detections by model for camera {folder} : {len(prediction[0])}")
		
		left_count = 0
		right_count = 0
		left_lane_detections=[]
		right_lane_detections= []

		if len(prediction) > 0:
			car_indices = torch.isin(prediction[0]['labels'], vehicle_labels_tensor) & (prediction[0]['scores'] > 0.5)
			car_boxes = prediction[0]['boxes'][car_indices].cpu().numpy()
			car_scores = prediction[0]['scores'][car_indices].cpu().numpy()
			car_labels = prediction[0]['labels'][car_indices].cpu().numpy()

			lane_start, lane_end = tuple(lane_2d_data[folder]["lane_point_1"]), tuple(lane_2d_data[folder]["lane_point_2"])

			# Process each detection
			for box, score, label in zip(car_boxes, car_scores, car_labels):
				x1, y1, x2, y2 = box
				cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

				# Assuming distance_to_line and point_relative_to_line are defined elsewhere
				dist = self.distance_to_line(cx, cy, *lane_start, *lane_end)
				position = self.point_relative_to_line(cx, cy, *lane_start, *lane_end)
				inSearchRegion = self.point_in_search_region(cx, cy, *lane_start, *lane_end)

				if dist < distance_threshold and inSearchRegion:
					if position == "right":
						right_count += 1
						right_lane_detections.append({"coordinates":[round(x1), round(y1), round(x2), round(y2)], "label":0})
					elif position == "left":
						left_count += 1
						left_lane_detections.append({"coordinates":[round(x1), round(y1), round(x2), round(y2)], "label":0})

		return [left_lane_detections, right_lane_detections]

	@classmethod
	def distance_to_line(cls, px, py, ax, ay, bx, by):
		"""Calculate the distance from a point P(px, py) to a line defined by points A(ax, ay) and B(bx, by)."""
		num = abs((bx - ax) * (ay - py) - (ax - px) * (by - ay))
		den = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
		return num / den

	@classmethod
	def point_relative_to_line(cls, px, py, ax, ay, bx, by):
		"""Determine if a point P(px, py) is left or right of a line AB. Returns 'left', 'right', or 'on'."""
		position = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
		if position > 0:
			return 'left'
		elif position < 0:
			return 'right'
		else:
			return 'on'
	
	def point_in_search_region(self, cx, cy, ax, ay, bx, by):
			# Compute the vectors C->B and A->B
			vector_CB = (bx - cx, by - cy)
			vector_AB = (bx - ax, by - ay)

			# Compute the dot product of vectors C->B and A->B
			dot_product = vector_CB[0] * vector_AB[0] + vector_CB[1] * vector_AB[1]
			if dot_product>0:
				return True
			return False