import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import os
import json
import argparse
from bounding_box import *
from cameras_list import *
from concurrent.futures import ThreadPoolExecutor
from traffic_light_manual_control import *

# Constants
fixed_delta_seconds = 0.08
save_after_every_n_frame = round(1/fixed_delta_seconds) # Fixed time step for simulation (13 for 1 FPS). 
time_threshold = 1000*round(1/fixed_delta_seconds) # To destroy stationary or frozen actors.
frame_count = 0
dict_traffic_id = {'A':17, 'B':15, 'C':23, 'D':16} # Indicates the actor id of the traffic signals in the junction.
json_file_path = "Intersection_camera.json" # Path to save simulation data
throughput = 0
total_queue_length = 0

# Load or create a JSON file to store camera data
try:
	with open(json_file_path, 'r') as file:
		# If the file exists, load the dictionary from it
		camera_json = json.load(file)
except FileNotFoundError:
	# If the file doesn't exist, create an empty dictionary
	camera_json = {}

n_last_camera_json = int(float(list(camera_json.keys())[-1])) + 1 if camera_json else 0 # Gets the last entry of the json file.
flag_camera_json = 1 # 1 indicates beginning of a new simulation.

class CameraParams:
	"""Class to store parameters for each camera"""
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
		"""Add a camera with custom parameters to the world"""
		blueprint_library = world.get_blueprint_library()
		camera_bp = blueprint_library.find('sensor.camera.rgb')
		camera_bp.set_attribute('image_size_x', str(self.image_size_x))
		camera_bp.set_attribute('image_size_y', str(self.image_size_y))
		camera_bp.set_attribute('fov', str(self.fov))
		camera_transform = carla.Transform(self.location, self.rotation)
		camera_sensor = world.spawn_actor(camera_bp, camera_transform)
		self.camera = camera_sensor
		
	def build_projection_matrix(self):
		"""Build the camera projection matrix"""
		focal = self.image_size_x / (2.0 * np.tan(float(self.fov) * np.pi / 360.0))
		K = np.identity(3)
		K[0, 0] = K[1, 1] = focal
		K[0, 2] = int(self.image_size_x) / 2.0
		K[1, 2] = int(self.image_size_y) / 2.0
		self.K = K
			
# Setup simulation
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

#Create CameraParams instances based on the provided parameters and add them to a list.
def create_camera_params(world, camera_params):
	camera_list = []
	for params in camera_params:
		camera_params_instance = CameraParams(**params)
		
		camera_params_instance.add_camera(world)
		camera_params_instance.build_projection_matrix()
		camera_params_instance.queue = queue.Queue()

		camera_list.append(camera_params_instance)	
	return camera_list

#spawn vehicles in the map
def spawn_npcs(world, traffic_manager, bp_lib, spawn_points, num_vehicles):
	bikes = ['vehicle.harley-davidson.low_rider', 'vehicle.vespa.zx125', 'vehicle.bh.crossbike', 'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
	available_vehicle_bps = [bp for bp in bp_lib.filter('vehicle') if bp.id not in bikes]
		
	actor_to_destroy = []
	for _ in range(num_vehicles):
		vehicle_bp = random.choice(available_vehicle_bps)
		
		spawn_point = random.choice([spawn_points[126], spawn_points[135], spawn_points[130], spawn_points[34], spawn_points[80], spawn_points[85], spawn_points[47], spawn_points[48]])
		npc = world.try_spawn_actor(vehicle_bp, spawn_point)

		if npc:
			npc.set_autopilot(True)
			all_vehicles_travel_time[npc.id] = TravelTime(npc.id)
			actor_to_destroy.append(npc.id)

	return actor_to_destroy

#Create camera listeners for each camera in the given list.
def create_camera_listeners(cameras):
	camera_listener = [None] * len(cameras)
	for i, camera_params in enumerate(cameras):
		camera_listener[i] = camera_params.camera.listen(camera_params.queue.put)
	return camera_listener

def save_frames_and_update_json(frame_count, cameras, camera_json, dict_traffic_id, customTrafficLight, store_interval):
	"""Save frames from each camera to disk and update the JSON data with relevant information."""

	global flag_camera_json
	n_last_camera_json = int(float(list(camera_json.keys())[-1])) + 1 if camera_json else 0
	
	# Iterate over each camera and process the captured images
	for i, camera_params in enumerate(cameras):
		image = camera_params.queue.get()
		img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

		# Perform lane detection and get relevant information
		lane_coordinates = camera_params.lane_coordinates
		K = camera_params.K
		# Get the transformation matrix from world to camera
		world_2_camera = np.array(camera_params.camera.get_transform().get_inverse_matrix())
		
		# Obtain lane detection by ground truth carla detection.
		left_lane_detections, right_lane_detections, npc_ids = LaneDetection.return_lane_detections(world, camera_params.camera, K, world_2_camera, lane_coordinates)

		# Draw left lane detections on the image
		for detection in left_lane_detections:
			(x_min, y_min, x_max, y_max) = detection['coordinates']
			label = detection["label"]
			if (label == 0): # label 0 means stopped
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
			
		# Draw right lane detections on the image
		for detection in right_lane_detections:
			(x_min, y_min, x_max, y_max) = detection['coordinates']
			label = detection["label"]
			if (label == 0): # label 0 means stopped
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
		
		if cv2.waitKey(1) == ord('q'):
			break

		# Disable vehicle Traveltime parameter for those vehicles who had earlier exited the city in destroy exit actor function.
		for npc_id in npc_ids:
			all_vehicles_travel_time[npc_id].exit_reached = False

		n_frame = save_after_every_n_frame*store_interval
			
		 # Save frames after every n_frame and update JSON data
		if frame_count % n_frame == 0 and frame_count>60:
			folder = 'Unknown'
			if i in range(4):
				folder = chr(ord('A') + i)
			
			image_path = os.path.abspath(os.path.join(folder, f'{n_last_camera_json}.jpg'))
			image.save_to_disk(image_path)

			camera_json.setdefault(n_last_camera_json, {'flag':flag_camera_json})
			flag_camera_json = 0

			camera_json[n_last_camera_json]["Throughput"] = throughput
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
				"Individual_camera_reward":{
					"waiting_time": round(sum(all_vehicles_wait_time[npc_id].get_wait_time() for npc_id in npc_ids)), 
					"queue_length": (len(left_lane_detections)+len(right_lane_detections))
					}
			}

		#Display text on image.
		dictionary = {"waiting_time": round(sum(all_vehicles_wait_time[npc_id].get_wait_time() for npc_id in npc_ids)), 
					"queue_length": (len(left_lane_detections)+len(right_lane_detections)),
					"throughput":throughput}
		img = LaneDetection.display_metric_on_image(img, dictionary)
		output_image_array.append(img)
	return output_image_array


# Dictionary to store the last observed location of each actor
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
						stationary_time = time.time() - last_location[actor.id]["last_update_time"]
						if stationary_time > time_threshold:
							print(f"Destroying actor {actor.id} - Stationary for {stationary_time} seconds")
							print(f"length before {len(actors_to_destroy)}")
							actors_to_destroy.remove(actor.id)
							actor.destroy()
							print(f"length after {len(actors_to_destroy)}")

	except KeyboardInterrupt:
		print("Script interrupted by user.")

	return actors_to_destroy

def destroy_actors_exit_city(world, spawn_points, actors_to_destroy):
	global throughput
	# Get all actors in the world
	actors = world.get_actors()

	#Destroy vehicle if exits the city.
	exit_locations = [26, 77, 67, 55]
	threshold_distance = 5
	try:
		# Iterate over actors and check if they are stationary for more than the threshold
		for actor in actors:
			if actor.type_id.startswith('vehicle'):
				current_location = actor.get_location()
				#Destroy vehicle if exits the city.
				for exit_location in exit_locations:
					spawn_location = spawn_points[exit_location].location
					distance = math.sqrt((current_location.x - spawn_location.x) ** 2 + (current_location.y - spawn_location.y) ** 2 + (current_location.z - spawn_location.z) ** 2)
					if distance < threshold_distance and all_vehicles_travel_time[actor.id].exit_reached == False:
						all_vehicles_travel_time[actor.id].stop_timer()
						all_vehicles_travel_time[actor.id].exit_reached = True
						all_vehicles_wait_time[actor.id].wait_time = 0
						if exit_location == 26:
							actors_to_destroy.remove(actor.id)
							actor.destroy()
						throughput += 1
						break
	except KeyboardInterrupt:
		print("Script interrupted by user.")

	return actors_to_destroy

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Dataset Creation")
	# Define arguments
	parser.add_argument(
		"--total_vehicles", 
		type=int, 
		default=100, 
		help="Total number of vehicles to spawn in the simulation"
	)

	parser.add_argument(
		"--spawn_interval", 
		type=int, 
		default=5, 
		help="Interval (in seconds) at which a new vehicle is spawned in the simulation"
	)

	parser.add_argument(
		"--store_interval", 
		type=int, 
		default=1, 
		help="Interval (in seconds) at which simulation data is captured and stored"
	)
	args = parser.parse_args()

	# initial setup
	client, world, bp_lib, spawn_points = setup_simulation()

	# initialize traffic state 
	customTrafficLight = CustomTrafficLight(world)

	# Create camera parameters and listeners
	cameras = create_camera_params(world, camera_params_list)
	camera_listeners = create_camera_listeners(cameras)

	# Initialize time tracking variables
	start_time = time.time()
	last_update_check_time = time.time()
	spawn_update_time = time.time()
	last_exit_check_time = time.time()
	
	# Set traffic manager to synchronous mode
	traffic_manager = client.get_trafficmanager()
	traffic_manager.set_synchronous_mode(True)

	actors_to_destroy = [] # List to keep a track of all vehicles spawned.

	try:
		while True:
			# Advance the simulation by one tick
			world.tick()
			frame_count += 1

			output_image_array = []
			
			# spawn vehicles every 5 seconds such that total vehicles in the map are less than 100.
			if time.time() - spawn_update_time >= args.spawn_interval and len(actors_to_destroy)<args.total_vehicles: 
				actors_to_destroy.extend(spawn_npcs(world, traffic_manager, bp_lib, spawn_points, num_vehicles=1))
				spawn_update_time = time.time()

			# Save frames and update JSON data
			output_image_array = save_frames_and_update_json(frame_count, cameras, camera_json, dict_traffic_id, customTrafficLight, args.store_interval)
			LaneDetection.display_output(output_image_array)
			
			# Calculate and display FPS
			elapsed_time = time.time() - start_time
			fps = frame_count / elapsed_time

			#Destroy the actors that exit the city
			if(time.time()-last_exit_check_time>0.5):
				actors_to_destroy = destroy_actors_exit_city(world, spawn_points, actors_to_destroy)
				last_exit_check_time = time.time()
			
			# Destroy stationary actors if actor not responding or actor stuck in traffic congestion.
			if(time.time()-last_update_check_time>5):
				actors_to_destroy = destroy_stationary_actors(world, spawn_points, actors_to_destroy, time_threshold)
				last_update_check_time = time.time()
				
			# Save camera data to JSON file every 200 seconds
			if (round(elapsed_time)%200 == 0):
				with open(json_file_path, 'w') as json_file:
					json.dump(camera_json, json_file, indent=4)

			 # Break simulation after 1800 seconds
			elapsed_time = round(elapsed_time)
			if (elapsed_time>0 and elapsed_time%1800==0):
				print("Time Duration Reached")
				break

	except KeyboardInterrupt:
		print("User interrupted the simulation.")
	finally:
		# Cleanup actors and sensors
		for i in range(len(cameras)):
			if camera_listeners[i] is not None:
				camera_listeners[i].stop()
			if cameras[i].camera.is_alive:
				cameras[i].camera.destroy()
			cameras[i].queue = queue.Queue()

		time.sleep(2)
		print("Destroying all actors and sensors.")
		for actor in world.get_actors():
				if actor.id in actors_to_destroy:
					if actor is not None and actor.is_alive:
						try:
							actor.destroy()
						except Exception as e:
							print(f"Failed to destroy actor {actor.id}: {e}")
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
			vehicle_wait_time_dict[id] = waitclass.total_wait_time
			TWT += vehicle_wait_time_dict[id]
		with open('waiting_time_file', 'w') as json_file:
			json.dump(vehicle_wait_time_dict, json_file, indent=4)
			print("wait time json updated")

		# analysis of the simulation
		print(f"Duration: {elapsed_time}")
		print(f"Total wait time: {TWT}")
		print(f"Total queue length: {total_queue_length}")
		print(f"Average travel time: {TravelTime.avg_travel_time()}")
		print(f"Throughput: {TravelTime.destination_reached}")