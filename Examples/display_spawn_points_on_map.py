import carla
import time

# Initialize the CARLA client and connect to the server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)  # Set a timeout for the connection

# Get the world object
world = client.get_world()
distance = 2

# Display spawn points on map for a certain period given by lifetime.
spawn_points = world.get_map().get_spawn_points()
for i, spawn_point in enumerate(spawn_points):
    world.debug.draw_string(spawn_point.location, str(i), life_time=500)
