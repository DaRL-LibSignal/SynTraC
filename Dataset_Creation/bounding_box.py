import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time

class CameraProjection:
	@staticmethod
	def get_image_point(loc, K, w2c):
		# Project 3D coordinates to 2D using the camera projection matrix
		point = np.array([loc.x, loc.y, loc.z, 1])
		point_camera = np.dot(w2c, point)

		# Convert from UE4's coordinate system to a standard one
		point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

		# Project 3D to 2D using the camera matrix
		point_img = np.dot(K, point_camera)
		point_img[0] /= point_img[2]
		point_img[1] /= point_img[2]

		return point_img[0:2]

class Vector3D:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	@staticmethod
	def normalize_vector(vector):
		length = math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
		if length == 0:
			return Vector3D(0, 0, 0)
		else:
			return Vector3D(vector.x / length, vector.y / length, vector.z / length)

	@staticmethod
	def dot_product(vector1, vector2):
		return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

all_vehicles_wait_time = {}
class WaitTime():
	def __init__(self, id):
		self.start_time = -1
		self.wait_time = 0
		self.total_wait_time = 0
		self.id = id
	def start_timer(self):
		if self.start_time == -1:
			self.start_time = time.time()

		else:
			self.wait_time = max(self.wait_time, time.time()-self.start_time)
	def stop_timer(self):
		if self.start_time != -1:

			self.start_time = -1
			self.total_wait_time += self.wait_time

	def get_wait_time(self):
		return self.wait_time
			
all_vehicles_travel_time = {}
class TravelTime():
	total_travel_time = 0
	destination_reached = 0
	def __init__(self, id):
		self.start_time = time.time()
		self.travel_time = 0
		self.id = id
		self.exit_reached = False

	def stop_timer(self):
		self.travel_time = time.time()-self.start_time
		TravelTime.total_travel_time += self.travel_time
		TravelTime.destination_reached += 1

	@classmethod
	def avg_travel_time(cls):

		return TravelTime.total_travel_time/TravelTime.destination_reached
			
#Calculate queue length
def calculate_queue_length(data):
	for i in range(4):
		folder = chr(ord('A') + i)
		TV = data[folder]["Lane1"]["total_vehicles"]+data[folder]["Lane2"]["total_vehicles"]
	return TV		
		
class LaneDetection:
	
	@staticmethod
	def get_lane_direction_vector(lane_point1, lane_point2):
		# Calculate the direction vector of a lane
		return Vector3D(lane_point2.x - lane_point1.x, lane_point2.y - lane_point1.y, lane_point2.z - lane_point1.z)

	@staticmethod
	def is_vehicle_on_right_of_lane(vehicle_centroid, lane_point1, lane_point2):
		# Check if a vehicle is on the left or right of a lane
		lane_vector = LaneDetection.get_lane_direction_vector(lane_point1, lane_point2)
		vehicle_vector = Vector3D(vehicle_centroid.x - lane_point1.x, vehicle_centroid.y - lane_point1.y, vehicle_centroid.z - lane_point1.z)
		# cross product between a vector formed by the lane coordinates and a vector connecting the vehicle centroid to one of the lane coordinates.
		cross_product = Vector3D(
			lane_vector.y * vehicle_vector.z - lane_vector.z * vehicle_vector.y,
			lane_vector.z * vehicle_vector.x - lane_vector.x * vehicle_vector.z,
			lane_vector.x * vehicle_vector.y - lane_vector.y * vehicle_vector.x
		)

		if cross_product.z < 0:
			return True
		else:
			return False

	@staticmethod
	def return_lane_detections(world, camera, K, world_2_camera, lane_coordinates):
		# Return lane detections based on given parameters
		initial_transform = lane_coordinates["initial_transform"]
		final_transform = lane_coordinates["final_transform"]
		ref_vector = initial_transform.location - camera.get_transform().location
		normalized_ref_vector = Vector3D.normalize_vector(ref_vector)

		left_lane_detections = []
		right_lane_detections = []
		npc_ids = []
		

		for npc in world.get_actors().filter('*vehicle*'):
			if npc.id not in all_vehicles_wait_time.keys():
				all_vehicles_wait_time[npc.id] = WaitTime(npc.id)
			bb = npc.bounding_box
			dist = npc.get_transform().location.distance(camera.get_transform().location)
			#we are restricting our search region between 10 and 60 meters from the traffic signal.
			if 10 < dist < 60:
				forward_vec = npc.get_transform().get_forward_vector()
				normalized_forward_vector = Vector3D.normalize_vector(forward_vec)
				ray = npc.get_transform().location - camera.get_transform().location
				normalized_ray = Vector3D.normalize_vector(ray)

				dot = Vector3D.dot_product(normalized_forward_vector, normalized_ray)
				dot_ref = Vector3D.dot_product(normalized_ref_vector, normalized_ray)
				point_right = LaneDetection.is_vehicle_on_right_of_lane(npc.get_transform().location, initial_transform.location, final_transform.location)
				
				vehicle_velocity = npc.get_velocity()
				forward_speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)

				#dot checks if the vehicle is front of the camera and not behind the camera.
				#dot_ref constraints the search region to only incoming traffic and not outgoing traffic.
				#point_right helps to get the right lane detections in front of the camera.
				if dot < -0.78 and dot_ref > 0.85 and point_right:
					verts = [v for v in bb.get_world_vertices(npc.get_transform())]
					x_max, x_min, y_max, y_min = -10000, 10000, -10000, 10000

					for vert in verts:
						p = CameraProjection.get_image_point(vert, K, world_2_camera)
						x_max = max(x_max, p[0])
						x_min = min(x_min, p[0])
						y_max = max(y_max, p[1])
						y_min = min(y_min, p[1])
					if forward_speed <= 0.1:
						moving = 0
						all_vehicles_wait_time[npc.id].start_timer()
					else:
						moving = 1
						all_vehicles_wait_time[npc.id].stop_timer()

					npc_ids.append(npc.id)							
					right_lane_detections.append({"coordinates":[round(x_min), round(y_min), round(x_max), round(y_max)], "label":moving})

				elif dot < -0.8 and dot_ref > 0.85 and not point_right:
					verts = [v for v in bb.get_world_vertices(npc.get_transform())]
					x_max, x_min, y_max, y_min = -10000, 10000, -10000, 10000

					for vert in verts:
						p = CameraProjection.get_image_point(vert, K, world_2_camera)
						x_max = max(x_max, p[0])
						x_min = min(x_min, p[0])
						y_max = max(y_max, p[1])
						y_min = min(y_min, p[1])
					if forward_speed <= 0.1:
						moving = 0
						all_vehicles_wait_time[npc.id].start_timer()
					else:
						moving = 1
						all_vehicles_wait_time[npc.id].stop_timer()

					npc_ids.append(npc.id)
					left_lane_detections.append({"coordinates":[round(x_min), round(y_min), round(x_max), round(y_max)], "label":moving})

		return left_lane_detections, right_lane_detections, npc_ids

	@staticmethod
	def display_output(output_image_array):
		# Display the output images of all 4 cameras in a single grid
		image1, image2, image3, image4 = output_image_array[0], output_image_array[1], output_image_array[2], output_image_array[3]
		height, width = 530, 1300
		image1 = cv2.resize(image1, (width, height))
		image2 = cv2.resize(image2, (width, height))
		image3 = cv2.resize(image3, (width, height))
		image4 = cv2.resize(image4, (width, height))

		top_row = np.hstack((image1, image2))
		bottom_row = np.hstack((image3, image4))
		grid_image = np.vstack((top_row, bottom_row))

		cv2.imshow("Image Grid", grid_image)
		return grid_image
	
	@staticmethod
	def display_metric_on_image(img, dictionary):
		font = cv2.FONT_HERSHEY_SIMPLEX 
		org = (50, 50) 
		fontScale = 1
		color = (0, 0, 255) 
		thickness = 3
		# Convert dictionary to string with each key-value pair on a separate line
		y_offset = org[1]  # Initial y-coordinate of the text
		for key, value in dictionary.items():
			text = f"{key}: {value}"
			img = cv2.putText(img, text, (org[0], y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
			y_offset += 30
		return img
