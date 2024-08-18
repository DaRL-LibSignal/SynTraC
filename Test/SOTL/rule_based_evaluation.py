import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import os
import json
from bounding_box import *
from cameras_list import *
import torch
from concurrent.futures import ThreadPoolExecutor
from traffic_light_manual_control import *

# Constants
fixed_delta_seconds = 0.08
save_after_every_n_frame = round(1/fixed_delta_seconds) # 13  For 1 FPS 
time_threshold = 1000*round(1/fixed_delta_seconds)
frame_count = 0
dict_traffic_id = {'A':17, 'B':15, 'C':23, 'D':16}
json_file_path = "Intersection_camera.json"
total_queue_length = 0

try:
	with open(json_file_path, 'r') as file:
		# If the file exists, load the dictionary from it
		camera_json = json.load(file)
except FileNotFoundError:
	# If the file doesn't exist, create an empty dictionary
	camera_json = {}
n_last_camera_json = int(float(list(camera_json.keys())[-1])) + 1 if camera_json else 0
flag_camera_json = 1

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
		
	# Function to add a camera with custom parameters
	def add_camera(self, world):
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

def create_camera_params(world, camera_params):
	camera_list = []
	for params in camera_params:
		camera_params_instance = CameraParams(**params)
		
		camera_params_instance.add_camera(world)
		camera_params_instance.build_projection_matrix()
		camera_params_instance.queue = queue.Queue()

		camera_list.append(camera_params_instance)
		
	return camera_list


def spawn_npcs(world, traffic_manager, bp_lib, spawn_points, route_set, spawn_cord, num_vehicles):
	bikes = ['vehicle.harley-davidson.low_rider', 'vehicle.vespa.zx125', 'vehicle.bh.crossbike', 'vehicle.kawasaki.ninja', 'vehicle.yamaha.yzf', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
	available_vehicle_bps = [bp for bp in bp_lib.filter('vehicle') if bp.id not in bikes]
		
	actor_to_destroy = []
	for _ in range(num_vehicles):
		vehicle_bp = random.choice(available_vehicle_bps)
		
		# spawn_point = carla.Transform(carla.Location(x=24.370887756347656,y=13.416457176208496,z=1.8430284261703491), carla.Rotation(pitch=-4.684423446655273,yaw=-174.84170532226562,roll=0.0001189384274766780))
		# spawn_point = random.choice([spawn_points[78], spawn_points[106], spawn_points[126], spawn_points[135]])
		# spawn_point = random.choice([spawn_points[126], spawn_points[135], spawn_points[130]])
		# npc = world.try_spawn_actor(vehicle_bp,random.choice(spawn_points))

		spawn_point = spawn_points[spawn_cord[route_set%len(spawn_cord)]]
		npc = world.try_spawn_actor(vehicle_bp, spawn_point)
		# traffic_manager.set_path(npc, route_1)
		if npc:
			npc.set_autopilot(True)
			all_vehicles_travel_time[npc.id] = TravelTime(npc.id)
			actor_to_destroy.append(npc.id)

	return actor_to_destroy

def create_camera_listeners(cameras):
	camera_listener = [None] * len(cameras)
	for i, camera_params in enumerate(cameras):
		camera_listener[i] = camera_params.camera.listen(camera_params.queue.put)
	return camera_listener

def display_camera_output(cameras):
	for i, camera_params in enumerate(cameras):
		image = camera_params.queue.get()
		img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
		cv2.namedWindow(str(i), cv2.WINDOW_AUTOSIZE)
		cv2.imshow(str(i), img)
		cv2.waitKey(1)


def save_frames_and_update_json(frame_count, cameras, camera_json, dict_traffic_id, customTrafficLight):
	global flag_camera_json
	global total_queue_length
	n_last_camera_json = int(float(list(camera_json.keys())[-1])) + 1 if camera_json else 0
	
	for i, camera_params in enumerate(cameras):
		image = camera_params.queue.get()
		# print(f"queue length for camera{i} is {camera_params.queue.qsize()}")
		img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
		lane_coordinates = camera_params.lane_coordinates
		K = camera_params.K
		world_2_camera = np.array(camera_params.camera.get_transform().get_inverse_matrix())
		left_lane_detections, right_lane_detections = LaneDetection.return_lane_detections(world, camera_params.camera, K, world_2_camera, lane_coordinates)
		# print(f"right_lane_detections {right_lane_detections}")
	
		for detection in left_lane_detections:
			(x_min, y_min, x_max, y_max) = detection['coordinates']
			label = detection["label"]
			if (label == 0):
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
			
		for detection in right_lane_detections:
			(x_min, y_min, x_max, y_max) = detection['coordinates']
			label = detection["label"]
			if (label == 0):
				cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 2)
				cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 2)
		
		if cv2.waitKey(1) == ord('q'):
			break

		output_image_array.append(img)

		n_frame = save_after_every_n_frame
		# print(f"frame_count{frame_count}")
			
		if frame_count % n_frame == 0 and frame_count>30:
			folder = 'Unknown'
			if i in range(4):
				folder = chr(ord('A') + i)
			
			image_path = os.path.abspath(os.path.join(folder, f'{n_last_camera_json}.jpg'))
			# image.save_to_disk(image_path)

			camera_json.setdefault(n_last_camera_json, {'flag':flag_camera_json})
			flag_camera_json = 0
			# Label 0 means stopped
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
		#Rule based function
		accumulated_vehicles = sum(1 for entry in left_lane_detections if entry["label"] == 0) + sum(1 for entry in right_lane_detections if entry["label"] == 0)
		customTrafficLight.sotl_check(accumulated_vehicles)

	# print(f"frame count {frame_count}")
	if customTrafficLight.check_RL_action_time():
		#Used to set the traffic light back to default value.
		customTrafficLight.initialize_traffic_state(world)
		customTrafficLight.reset_RL_action_time()
		#calculate the queue length
		total_queue_length += calculate_queue_length(camera_json[list(camera_json.keys())[-1]])

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
					if distance < threshold_distance:
						# print(f"Destroying actor {actor.id} near spawn point.")
						# print(f"length before {len(actors_to_destroy)}")
						actors_to_destroy.remove(actor.id)
						all_vehicles_travel_time[actor.id].stop_timer()
						actor.destroy()
						# print(f"length after {len(actors_to_destroy)}")
						break
	except KeyboardInterrupt:
		print("Script interrupted by user.")

	return actors_to_destroy

# Main script
if __name__ == "__main__":
	customTrafficLight = CustomTrafficLight(carla.Client('localhost', 2000).get_world())
	print("Initial condition set")

	#setting up the simulation
	client, world, bp_lib, spawn_points = setup_simulation()

	cameras = create_camera_params(world, camera_params_list)
	camera_listeners = create_camera_listeners(cameras)

	start_time = time.time()
	last_update_check_time = time.time()
	last_exit_check_time = time.time()
	spawn_update_time = time.time()
	
	
	traffic_manager = client.get_trafficmanager()
	traffic_manager.set_synchronous_mode(True)

	actors_to_destroy = []
	route_set=0

	try:
		while True:
			# print(f"Len actors {len(actor_to_destroy)}")
			world.tick()
			frame_count += 1
			# print(f"frame_count {frame_count}")
			output_image_array = []
			elapsed_time = time.time() - start_time
			
			# actors_to_destroy.extend(spawn_npcs(world, traffic_manager, bp_lib, spawn_points, num_vehicles=100))
			if time.time() - spawn_update_time >= 1 and len(actors_to_destroy)<80:  # Check if 3 seconds have passed
				if elapsed_time<40:
					spawn_cord = spawn_cord = {0:126, 1:135}
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

			# display_camera_output(cameras)
			output_image_array = save_frames_and_update_json(frame_count, cameras, camera_json, dict_traffic_id, customTrafficLight)
			LaneDetection.display_output(output_image_array)
			
			fps = frame_count / elapsed_time
			# print(f"FPS: {fps:.2f}")
			
			#Destroy stationary actors
			if(time.time()-last_update_check_time>5):
				actors_to_destroy = destroy_stationary_actors(world, spawn_points, actors_to_destroy, time_threshold)
				last_update_check_time = time.time()

			#Destroy the actors that exit the city
			if(time.time()-last_exit_check_time>0.5):
				actors_to_destroy = destroy_actors_exit_city(world, spawn_points, actors_to_destroy)
				last_exit_check_time = time.time()

			if (round(elapsed_time)%200 == 0):
				with open(json_file_path, 'w') as json_file:
					json.dump(camera_json, json_file, indent=4)
					# print("json updated")

			elapsed_time = round(elapsed_time)
			if (elapsed_time>0 and elapsed_time%200==0):
				print("Time Duration Reached")
				break

	except KeyboardInterrupt:
		print("User interrupted the simulation.")
	finally:
		customTrafficLight.reset_traffic_state()
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
			vehicle_wait_time_dict[id] = waitclass.get_wait_time()
			TWT += vehicle_wait_time_dict[id]
		# with open('waiting_time_file', 'w') as json_file:
		# 	json.dump(vehicle_wait_time_dict, json_file, indent=4)
			# print("wait time json updated")
		print(f"Duration: {elapsed_time}")
		print(f"Total wait time: {TWT}")
		print(f"Total queue length: {total_queue_length}")
		print(f"Average travel time: {TravelTime.avg_travel_time()}")
		print(f"Throughput: {TravelTime.destination_reached}")