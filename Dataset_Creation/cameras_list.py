import carla

# Define camera parameters
camera_params_list = [
    {
        'image_size_x': 1920,
        'image_size_y': 1080,
        'fov': 90,
        'location': carla.Location(x=-52.55995178222656, y=20.053571701049805, z=5.32501745223999),
        'rotation': carla.Rotation(pitch=2.9262304306030273, yaw=-77.59104919433594, roll=-0.015502972528338432),
		'lane_coordinates' : {
				'initial_transform': carla.Transform(carla.Location(x=-50.5479621887207, y=-0.06838958710432053, z=0.7329870462417603)),
				'final_transform': carla.Transform(carla.Location(x=-50.52410888671875, y=-37.14356994628906, z=0.8263704776763916))
			}
    },
    {
        'image_size_x': 1920,
        'image_size_y': 1080,
        'fov': 90,
        'location': carla.Location(x=-41.78554916381836, y=20.318445205688477, z=5.50205659866333),
        'rotation': carla.Rotation(pitch=-0.20571845769882202, yaw=99.57952880859375, roll=-0.015502914786338806),
		'lane_coordinates' : {
				'initial_transform': carla.Transform(carla.Location(x=-43.33644485473633, y=41.12093734741211, z=1.0258973836898804)),
				'final_transform': carla.Transform(carla.Location(x=-43.38214874267578, y=101.14701843261719, z=1.3875439167022705))
			}
    },
    {
        'image_size_x': 1920,
        'image_size_y': 1080,
        'fov': 90,
        'location': carla.Location(x=-31.72666358947754,y=25.14920425415039,z=5.3892974853515625),
        'rotation': carla.Rotation(pitch=-26.11623764038086,yaw=-177.64808654785156,roll=-0.01434326171875),
		'lane_coordinates' : {
				'initial_transform': carla.Transform(carla.Location(x=-64.63727569580078, y=26.18655014038086, z=0.9430992007255554)),
				'final_transform': carla.Transform(carla.Location(x=-90.22425842285156, y=26.105587005615234, z=0.680543065071106))
			}
    },
    {
        'image_size_x': 1920,
        'image_size_y': 1080,
        'fov': 90,
        'location': carla.Location(x=-64.12722778320312,y=13.716440200805664,z=5.444387912750244),
        'rotation': carla.Rotation(pitch=-27.55165672302246, yaw=-0.4765319228172302, roll=-0.014282218180596828),
		'lane_coordinates' : {
				'initial_transform': carla.Transform(carla.Location(x=-30.52121353149414, y=14.846488952636719, z=0.6784976124763489)),
				'final_transform': carla.Transform(carla.Location(x=16.63819694519043, y=15.001932144165039, z=1.2565994262695312))
			}
    }
    # Add more camera parameters as needed
]