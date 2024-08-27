'''
This script is used to run the entire simulation on carla. Data is captured from the junction cameras
and analytics is performed on them.
This drives the RL model which is integrated inside this code and outputs whether the state of traffic lights
should change or not. The traffic lights are also connected in this code and act accordingly to the 
output as given by the RL model. Performance of this RL model is noted at the end of the time duration.
'''

import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import os
import argparse
import json
from Core.bounding_box import *
from Core.cameras_list import *
import torch
from concurrent.futures import ThreadPoolExecutor
from Core.traffic_light_manual_control import *
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn, ssdlite320_mobilenet_v3_large
import torchvision.transforms as transforms
from PIL import Image
from Core.object_detection_model import Object_Detection_Model

# Constants
fixed_delta_seconds = 0.08
save_after_every_n_frame = round(1/fixed_delta_seconds) # Fixed time step for simulation (13 for 1 FPS). 
time_threshold = 1000*round(1/fixed_delta_seconds) # To destroy stationary or frozen actors.
frame_count = 0
dict_traffic_id = {'A':17, 'B':15, 'C':23, 'D':16} # Indicates the actor id of the traffic signals in the junction.
json_file_path = "Intersection_camera.json" # Path to save simulation data
total_queue_length = 0

# Load or create a JSON file to store camera data
try:
	with open(json_file_path, 'r') as file:
		# If the file exists, load the dictionary from it
		camera_json = json.load(file)
except FileNotFoundError:
	# If the file doesn't exist, create an empty dictionary
	camera_json = {}
	with open(json_file_path, 'w') as json_file:
		json.dump(camera_json, json_file, indent=4)

lane_2d_data = {}

n_last_camera_json = int(float(list(camera_json.keys())[-1])) + 1 if camera_json else 0 # Gets the last entry of the json file.
flag_camera_json = 1 # 1 indicates beginning of a new simulation.

# Choose the object detection model.
object_model = Object_Detection_Model("RetinaNet")

class CameraParams:
	def __init__(self, image_size_x, image_size_y, fov, location, rotation, lane_coordinates):
		self.image_size_x = image_size_x
		self.image_size_y = image_size_y
		self.fov = fov
		self.location = location
		self.rotation = rotation
		self.lane_coordinates = lane_coordinates
		self.camera = None
		self.K = None
		self.queue = queue
		
	def add_camera(self, world):
		# Function to add a camera with custom parameters
		blueprint_library = world.get_blueprint_library()
		camera_bp = blueprint_library.find('sensor.camera.rgb')
		camera_bp.set_attribute('image_size_x', str(self.image_size_x))
		camera_bp.set_attribute('image_size_y', str(self.image_size_y))
		camera_bp.set_attribute('fov', str(self.fov))
		camera_transform = carla.Transform(self.location, self.rotation)
		camera_sensor = world.spawn_actor(camera_bp, camera_transform)
		self.camera = camera_sensor
		
	def build_projection_matrix(self):
		# Build the camera projection matrix
		focal = self.image_size_x / (2.0 * np.tan(float(self.fov) * np.pi / 360.0))
		K = np.identity(3)
		K[0, 0] = K[1, 1] = focal
		K[0, 2] = int(self.image_size_x) / 2.0
		K[1, 2] = int(self.image_size_y) / 2.0
		self.K = K
		

# Define RL Model for traffic light control
class RL_Model:
	# Initialize the policy with the provided policy path
	def __init__(self, policy_path):
		# Construct the full path to the policy file
		full_policy_path = os.path.join("policy_candidates", policy_path)
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.policy = torch.jit.load(full_policy_path, map_location=self.device)
		self.current_phase = 0
	
	def set_current_phase(self, x):
		self.current_phase = x
	
	def prepare_policy_input(self, input_json):
		count_torch = []
		for folder in ["A", "B", "C", "D"]:
			count_torch.append(input_json[folder]["Lane1"]["total_vehicles"])
			count_torch.append(input_json[folder]["Lane2"]["total_vehicles"])
		for i,folder in enumerate(["A", "B", "C", "D"]):
			if input_json[folder]["traffic_light"]=="Green":
				self.current_phase = i
				count_torch.append(self.current_phase)
		return torch.Tensor(count_torch)
			
	def get_model_output(self, torch_input, customTrafficLight):
		torch_input = (torch_input[None, :]).to(self.device)
		print(f"torch input {torch_input}")
		action = self.policy(torch_input)
		print(f"Action by policy {action} current phase {self.current_phase}")
		if action.item()!=self.current_phase:
			# print("changing state")
			customTrafficLight.change_next_cyclic_state()


# Setup the simulation	
def setup_simulation():
	client = carla.Client('localhost', 2000)
	world = client.get_world()
	bp_lib = world.get_blueprint_library()

	settings = world.get_settings()
	settings.synchronous_mode = True
	settings.fixed_delta_seconds = fixed_delta_seconds
	world.apply_settings(settings)

	spawn_points = world.get_map().get_spawn_points()
	return client, world, bp_lib, spawn_points

# Create camera parameters
def create_camera_params(world, camera_params):
	camera_list = []
	for params in camera_params:
		camera_params_instance = CameraParams(**params)
		
		camera_params_instance.add_camera(world)
		camera_params_instance.build_projection_matrix()
		camera_params_instance.queue = queue.Queue()

		camera_list.append(camera_params_instance)
	
	return camera_list

# Spawn vehicles function in carla to generate traffic.
def spawn_npcs(world, traffic_manager, bp_lib, spawn_points, route_set, spawn_cord, num_vehicles):
	bikes = ['vehicle.harley-davidson.low_rider', 'vehicle.vespa.zx125', 'vehicle.bh.crossbike', 'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
	available_vehicle_bps = [bp for bp in bp_lib.filter('vehicle') if bp.id not in bikes]
		
	actor_to_destroy = [] 
	for _ in range(num_vehicles):
		vehicle_bp = random.choice(available_vehicle_bps)

		spawn_point = spawn_points[spawn_cord[route_set%len(spawn_cord)]]
		npc = world.try_spawn_actor(vehicle_bp, spawn_point)
		if npc:
			npc.set_autopilot(True)
			all_vehicles_travel_time[npc.id] = TravelTime(npc.id)
			actor_to_destroy.append(npc.id)

	return actor_to_destroy

# Create camera listeners
def create_camera_listeners(cameras):
	camera_listener = [None] * len(cameras)
	for i, camera_params in enumerate(cameras):
		camera_listener[i] = camera_params.camera.listen(camera_params.queue.put)
	return camera_listener

# Display camera output
def display_camera_output(cameras):
	for i, camera_params in enumerate(cameras):
		image = camera_params.queue.get()
		img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
		cv2.namedWindow(str(i), cv2.WINDOW_AUTOSIZE)
		cv2.imshow(str(i), img)
		cv2.waitKey(1)


def save_frames_and_update_json(frame_count, cameras, camera_json, dict_traffic_id, customTrafficLight, rl_model):
	# Define global variables
	global flag_camera_json
	global total_queue_length

	# Determine the next key for the camera_json dictionary
	n_last_camera_json = int(float(list(camera_json.keys())[-1])) + 1 if camera_json else 0
	
	# Iterate over each camera and process the captured images
	for i, camera_params in enumerate(cameras):
		# Retrieve the image from the camera's queue
		image = camera_params.queue.get()
		# Reshape the image data
		img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
		# Extract lane coordinates and camera matrix
		lane_coordinates = camera_params.lane_coordinates
		K = camera_params.K
		# Get the transformation matrix from world to camera
		world_2_camera = np.array(camera_params.camera.get_transform().get_inverse_matrix())
		folder = chr(ord('A') + i)
		
		# Obtain lane detection by ground truth carla detection.
		left_lane_detections, right_lane_detections = LaneDetection.return_lane_detections(world, camera_params.camera, K, world_2_camera, lane_coordinates)
		
		# Obtain lane detections using object detection model on image.
		# left_lane_detections, right_lane_detections = object_model.process_image_for_detections(img, lane_2d_data, folder, distance_threshold=100)
	
		# Display bounding box of vehicles in left lane.
		for detection in left_lane_detections:
			(x_min, y_min, x_max, y_max) = detection['coordinates']
			label = detection["label"]
			if (label == 0):  # Red for stopped vehicles
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
			elif (label == 1): # Yellow for moving vehicles
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 255, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 255, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 255, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255, 255), 2)
			
		# Display bounding box of vehicles in right lane.
		for detection in right_lane_detections:
			(x_min, y_min, x_max, y_max) = detection['coordinates']
			label = detection["label"]
			if (label == 0): # Red for stopped vehicles
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
			elif (label == 1): # Yellow for moving vehicles
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 255, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 255, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 255, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255, 255), 2)
		
		# Break the loop if 'q' is pressed
		if cv2.waitKey(1) == ord('q'):
			break

		""" Storing the lane coordinates in 2d format. 
		We are projecting the 3d lane coordinates for each camera onto the image captured by the camera. 
		In ideal testing scenario we would just obtain the image from the camera, 
		thats why we aim to do all computation on the image ie 2d format."""
		if lane_2d_data.get(folder) is None:
			lane_point_1 = CameraProjection.get_image_point(lane_coordinates["initial_transform"].location, K, world_2_camera)
			lane_point_2 = CameraProjection.get_image_point(lane_coordinates["final_transform"].location, K, world_2_camera)
			lane_point_1 = tuple((int(lane_point_1[0]), int(lane_point_1[1])))
			lane_point_2 = tuple((int(lane_point_2[0]), int(lane_point_2[1])))
			lane_2d_data[folder] = {
				'lane_point_1': lane_point_1,
				'lane_point_2': lane_point_2
			}
		cv2.line(img, lane_2d_data[folder]['lane_point_1'], lane_2d_data[folder]['lane_point_2'], (0, 0, 255, 255), 2)
		output_image_array.append(img)

		# n_frame = save_after_every_n_frame
		n_frame = 10
			
		# Save frame and perform analysis on simulation data.
		if frame_count % n_frame == 0 and frame_count>30:
			
			image_path = os.path.abspath(os.path.join(folder, f'{n_last_camera_json}.jpg'))
			image.save_to_disk(image_path)

			camera_json.setdefault(n_last_camera_json, {'flag':flag_camera_json})
			flag_camera_json = 0
			# Label 0 means stopped and label 1 means vehicle is moving.
			camera_json[n_last_camera_json][folder] = {
				"image": image_path,
				"traffic_light": str(next((actor.get_state() for actor in world.get_actors() if actor.id == dict_traffic_id[folder]), None)),
				"timestamp" : time.time(),
				"Lane1":{
					"total_vehicles" : len(left_lane_detections),
					"stopped_vehicles": sum(1 for entry in left_lane_detections if entry["label"] == 0),
					"vehicles":left_lane_detections

				},
				"Lane2":{
					"total_vehicles" : len(right_lane_detections),
					"stopped_vehicles":sum(1 for entry in right_lane_detections if entry["label"] == 0),
					"vehicles":right_lane_detections
				},
				"Rewards":{
					"waiting_time":0, 
					"throughput":0, 
					"queue_length": -(len(left_lane_detections)+len(right_lane_detections))
					}
			}


	# This is the step where the RL model comes into action.
	if customTrafficLight.check_RL_action_time():
		# Sync the traffic state if it's time for RL action
		customTrafficLight.initialize_traffic_state(world)

		# Check if there are any entries in the camera_json
		if len(camera_json.keys())>0:
			# Prepare the input vector for the RL model
			input_vector = []

			# Loop through each camera (A, B, C, D)
			for x in range(4):
				folder = chr(ord('A') + x)
				image_path = camera_json[list(camera_json.keys())[-1]][folder]["image"]
				# print(image_path)

				# Process the image to get lane detections
				left_lane_detections, right_lane_detections = object_model.process_image_for_detections(image_path, None, lane_2d_data, folder, distance_threshold=100)
				
				# Append the number of detected vehicles in each lane to the input vector
				input_vector.append(len(left_lane_detections))
				input_vector.append(len(right_lane_detections))
			
			# Determine the current phase (i.e., which traffic light is green)
			for x in range(4):
				folder = chr(ord('A') + x)
				if camera_json[list(camera_json.keys())[-1]][folder]["traffic_light"]=="Green":
					rl_model.set_current_phase(x)
					input_vector.append(x)

			# Convert the input vector to a tensor
			torch_input = torch.Tensor(input_vector)
		
			# Get the RL model's output and update the traffic light state
			rl_model.get_model_output(torch_input, customTrafficLight)
			customTrafficLight.reset_RL_action_time()
		
		# Calculate and update the total queue length
		total_queue_length += calculate_queue_length(camera_json[list(camera_json.keys())[-1]])
	return output_image_array


# Dictionary to store the last observed location of each actor.
last_location = {}
def destroy_stationary_actors(world, spawn_points, actors_to_destroy, time_threshold):
	global last_location
	
	# Get all actors in the world
	actors = world.get_actors()

	try:
		# Iterate over actors and check if they are stationary for more than the threshold
		for actor in actors:
			if actor.type_id.startswith('vehicle'):
				current_location = actor.get_location()

				# Check if the actor's location has changed since the last observation
				if actor.id in last_location and last_location[actor.id]["location"] != current_location:
					last_location[actor.id] = {"location":current_location, "last_update_time":time.time()}
				else:
					# If the actor's location hasn't changed, update the time it has been stationary
					if actor.id not in last_location:
						last_location[actor.id] = {"location":current_location, "last_update_time":time.time()}
					else:
						# Calculate the time the actor has been stationary
						stationary_time = time.time() - last_location[actor.id]["last_update_time"]
						if stationary_time > time_threshold:
							# If the stationary time exceeds the threshold, destroy the actor
							print(f"Destroying actor {actor.id} - Stationary for {stationary_time} seconds")
							actors_to_destroy.remove(actor.id) # Remove the actor from the list of actors to destroy
							actor.destroy()  # Destroy the actor

	except KeyboardInterrupt:
		print("Script interrupted by user.")

	return actors_to_destroy

def destroy_actors_exit_city(world, spawn_points, actors_to_destroy):
	# Get all actors in the world
	actors = world.get_actors()

	# Define exit locations and the threshold distance for destroying actors
	exit_locations = [26, 77, 67, 55]
	threshold_distance = 5

	try:
		# Iterate over actors and check if they are at an exit location
		for actor in actors:
			if actor.type_id.startswith('vehicle'):
				current_location = actor.get_location()

				# Check if the vehicle is near any of the exit locations
				for exit_location in exit_locations:
					spawn_location = spawn_points[exit_location].location
					# Calculate the distance between the current location and the spawn location
					distance = math.sqrt((current_location.x - spawn_location.x) ** 2 + (current_location.y - spawn_location.y) ** 2 + (current_location.z - spawn_location.z) ** 2)
					# If the vehicle is within the threshold distance of an exit location, destroy it
					if distance < threshold_distance:
						actors_to_destroy.remove(actor.id)  # Remove the actor's ID from the list of actors to destroy
						all_vehicles_travel_time[actor.id].stop_timer()  # Stop the timer for the vehicle's travel time
						actor.destroy() # Destroy the actor
						break # Exit the loop since the actor has been destroyed
	except KeyboardInterrupt:
		print("Script interrupted by user.")

	return actors_to_destroy

# Main script
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="RL Model for Traffic Light Control")
	parser.add_argument("--policy_path", type=str, default='cnt_cql.pt', help="Path to the policy file (.pt)")

	args = parser.parse_args()

	customTrafficLight = CustomTrafficLight(carla.Client('localhost', 2000).get_world())
	print("Initial condition set")

	#setting up the simulation
	client, world, bp_lib, spawn_points = setup_simulation()

	# Create camera parameters and listeners
	cameras = create_camera_params(world, camera_params_list)
	camera_listeners = create_camera_listeners(cameras)

	# Initialize time tracking variables
	start_time = time.time()
	last_update_check_time = time.time()
	last_exit_check_time = time.time()
	spawn_update_time = time.time()
	
	# Set traffic manager to synchronous mode
	traffic_manager = client.get_trafficmanager()
	traffic_manager.set_synchronous_mode(True)

	actors_to_destroy = [] # List to keep a track of all vehicles spawned.
	route_set=0

	# Initialize the RL Model with the policy path from arguments
	rl_model = RL_Model(args.policy_path)

	try:
		while True:
			# Advance the simulation by one tick
			world.tick()
			frame_count += 1
			output_image_array = []
			elapsed_time = time.time() - start_time

			# Spawn vehicles if certain conditions are met
			if time.time() - spawn_update_time >= 3 and len(actors_to_destroy)<100:  # Check if 3 seconds have passed
				if elapsed_time<40:
					spawn_cord = {0:126, 1:135}
				elif elapsed_time<80:
					spawn_cord = {0:47, 1:48}
				elif elapsed_time<120:
					spawn_cord = {0:130, 1:34}
				elif elapsed_time<160:
					spawn_cord = {0:80, 1:85}
				else:
					spawn_cord = {0:80, 1:85, 2:47, 3:48}
				route_set += 1
				actors_to_destroy.extend(spawn_npcs(world, traffic_manager, bp_lib, spawn_points, route_set, spawn_cord, num_vehicles=1))
				spawn_update_time = time.time()

			# Save frames and update JSON data
			output_image_array = save_frames_and_update_json(frame_count, cameras, camera_json, dict_traffic_id, customTrafficLight, rl_model)
			LaneDetection.display_output(output_image_array)
			
			# Calculate and display FPS
			fps = frame_count / elapsed_time
			
			# Destroy stationary actors every 5 seconds
			if(time.time()-last_update_check_time>5):
				actors_to_destroy = destroy_stationary_actors(world, spawn_points, actors_to_destroy, time_threshold)
				last_update_check_time = time.time()

			# Destroy actors that exit the city every 0.5 seconds
			if(time.time()-last_exit_check_time>0.5):
				actors_to_destroy = destroy_actors_exit_city(world, spawn_points, actors_to_destroy)
				last_exit_check_time = time.time()

			# Periodically save JSON data
			if (round(elapsed_time)%200 == 0):
				with open(json_file_path, 'w') as json_file:
					json.dump(camera_json, json_file, indent=4)

			# Check if the simulation duration has been reached
			elapsed_time = round(elapsed_time)
			if (elapsed_time>0 and elapsed_time%200==0):
				print("Time Duration Reached")
				break

	except KeyboardInterrupt:
		print("User interrupted the simulation.")
	finally:
		# Reset the traffic light state
		customTrafficLight.reset_traffic_state()

		# Stop camera listeners and destroy cameras
		for i in range(len(cameras)):
			if camera_listeners[i] is not None:
				camera_listeners[i].stop()
			if cameras[i].camera.is_alive:
				cameras[i].camera.destroy()
			cameras[i].queue = queue.Queue()

		time.sleep(2)
		print("Destroying all actors and sensors.")
		# Destroy all actors in the list
		for actor in world.get_actors():
				if actor.id in actors_to_destroy:
					if actor is not None and actor.is_alive:
						try:
							actor.destroy()
						except Exception as e:
							print(f"Failed to destroy actor {actor.id}: {e}")

		# Revert world settings to asynchronous mode
		settings = world.get_settings()
		settings.synchronous_mode = False
		world.apply_settings(settings)
		cv2.destroyAllWindows()

		# Save the camera data to a JSON file
		with open(json_file_path, 'w') as json_file:
			json.dump(camera_json, json_file, indent=4)
			print("json updated")
			
		# Save the waiting data to a JSON file
		vehicle_wait_time_dict = {}
		TWT = 0
		for id, waitclass in all_vehicles_wait_time.items():
			vehicle_wait_time_dict[id] = waitclass.get_wait_time()
			TWT += vehicle_wait_time_dict[id]

		print(f"Duration: {elapsed_time}")
		print(f"Total wait time: {TWT}")
		print(f"Total queue length: {total_queue_length}")
		print(f"Average travel time: {TravelTime.avg_travel_time()}")
		print(f"Throughput: {TravelTime.destination_reached}")
