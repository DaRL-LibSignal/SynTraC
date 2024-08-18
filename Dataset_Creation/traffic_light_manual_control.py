'''
traffic lights id = [16,17,15,23]
Actor(id=16) :  D
Actor(id=17) :  A
Actor(id=15) :  B
Actor(id=23) :  C
'''

class CustomTrafficLight:

	def __init__(self, world):
		self.order = [17, 15, 23, 16]
		self.dict_traffic_id = {'A': 17, 'B': 15, 'C': 23, 'D': 16}
		self.group_TL = {}  # {16: carla.TL, 17: carla.TL}
		self.default_light_time = {"Red": 2, "Yellow": 3, "Green": 20}
		self.short_light_time = {"Red": 2, "Yellow": 3, "Green": 1}
		self.initialize_traffic_state(world)

	# For the first time we wait till the traffic light id 23 turns gree. This helps to set the same initial condition at the start.
	def initialize_traffic_state(self, world):
		if len(self.group_TL)==0:
			for actor in world.get_actors():
				if actor.id == 23:
					self.group_TL = {g.id: g for g in actor.get_group_traffic_lights()}
		for _, tl in self.group_TL.items():
			self.set_traffic_light_time(tl, self.default_light_time)

	def set_traffic_light_time(self, actor, light_time):
		actor.set_green_time(light_time["Green"])
		actor.set_red_time(light_time["Red"])
		actor.set_yellow_time(light_time["Yellow"])

	def print_traffic_light_times(self):
		for _, tl in self.group_TL.items():
			print(f"{_} Red {tl.get_red_time()} yellow {tl.get_yellow_time()} green {tl.get_green_time()}")

	def change_next_cyclic_state(self):
		for _, tl in self.group_TL.items():
			if tl.get_state().name != 'Red':
				self.set_traffic_light_time(tl, self.short_light_time)

	def reset_traffic_state(self):
		for _, tl in self.group_TL.items():
			self.set_traffic_light_time(tl, self.default_light_time)