import carla
import time

'''
traffic lights id = [16,17,15,23]
Actor(id=16) : [x=-64.264191, y=7.063309, z=0.254254] D
Actor(id=17) : [x=-62.352070, y=20.196909, z=0.254254] A
Actor(id=15) : [x=-31.930841, y=20.301954, z=0.254254] B
Actor(id=23) : [x=-31.636356, y=33.589287, z=0.254254] C
'''
def setup_simulation():
	client = carla.Client('localhost', 2000)
	world = client.get_world()
	bp_lib = world.get_blueprint_library()

	settings = world.get_settings()
	settings.synchronous_mode = True
	settings.fixed_delta_seconds = 0.08
	world.apply_settings(settings)

	spawn_points = world.get_map().get_spawn_points()

	return client, world

class CustomTrafficLight:

	def __init__(self, world):
		self.set_initial_condition = True
		self.order = [17, 15, 23, 16]
		self.dict_traffic_id = {'A': 17, 'B': 15, 'C': 23, 'D': 16}
		self.group_TL = {}  # {16: carla.TL, 17: carla.TL}
		self.min_light_time = {"Red": 2, "Yellow": 3, "Green": 10}
		self.default_light_time = {"Red": 2, "Yellow": 3, "Green": 10}
		self.max_light_time = {"Red": 2, "Yellow": 3, "Green": 30}
		self.short_light_time = {"Red": 2, "Yellow": 3, "Green": 0.01}
		self.RL_action_time = 0
		self.threshold_vehicle_sotl = 5
		self.initialize_traffic_state(world)

	def initialize_traffic_state(self, world):
		if self.set_initial_condition == True:
			if len(self.group_TL)==0:
				for actor in world.get_actors():
					if actor.id == 23:
						self.group_TL = {g.id: g for g in actor.get_group_traffic_lights()}
			#For same initial condition
			target_traffic_light_id = 16
			target_traffic_light = self.group_TL.get(target_traffic_light_id)
			if target_traffic_light:
				while True:
					state = target_traffic_light.get_state().name
					if state == 'Green':
						break
					time.sleep(0.1)  # Sleep briefly before checking again
			self.set_initial_condition = False
			self.RL_action_time = time.time()

		for _, tl in self.group_TL.items():
			self.set_traffic_light_time(tl, self.max_light_time)

	def set_traffic_light_time(self, actor, light_time):
		actor.set_green_time(light_time["Green"])
		actor.set_red_time(light_time["Red"])
		actor.set_yellow_time(light_time["Yellow"])

	def print_traffic_light_times(self):
		for _, tl in self.group_TL.items():
			print(f"{_} Red {tl.get_red_time()} yellow {tl.get_yellow_time()} green {tl.get_green_time()}")

	def change_next_cyclic_state(self):
		# print(self.group_TL.items())
		for _, tl in self.group_TL.items():
			if tl.get_state().name != 'Red':
				self.set_traffic_light_time(tl, self.short_light_time)
	
	def reset_traffic_state(self):
		for _, tl in self.group_TL.items():
			self.set_traffic_light_time(tl, self.default_light_time)

	def check_RL_action_time(self):
		if (time.time()-self.RL_action_time>20):
			return True
		return False
	def reset_RL_action_time(self):
		self.RL_action_time = time.time()

	def sotl_check(self, queue_length):
		#Change State
		#if current_green_time passed>min_green_time; queue length is greater than threshold; or time>greater than max_time
		for _, tl in self.group_TL.items():
			if tl.get_state().name == 'Green':
				if self.check_RL_action_time():
					print(f"Green Time {tl.get_green_time()} Elapsed Time {round(tl.get_elapsed_time())}")
				if (tl.get_elapsed_time() > self.min_light_time["Green"] and queue_length > self.threshold_vehicle_sotl):
					print(f"Before Changing Green Time {tl.get_green_time()} Elapsed Time {round(tl.get_elapsed_time())}")
					print("state changing")
					self.change_next_cyclic_state()
					print("state changed")


if __name__ == '__main__':
	client, world = setup_simulation()
	
	traffic_manager = client.get_trafficmanager()
	traffic_manager.set_synchronous_mode(True)

	# CustomTrafficLight.initialize_traffic_state(world)
	customTrafficLight = CustomTrafficLight(world)
	customTrafficLight.print_traffic_light_times()
	customTrafficLight.change_next_cyclic_state()