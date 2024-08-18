import torch

BATCH_SIZE = 12 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 5 # Number of epochs to train for.
NUM_WORKERS = 0 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_DIR =  "/home/local/ASURITE/tchen169/Documents/CV4TSC/rough/RL_Model/cyclic_traffic_time_for_train/1fpsDataset/"
# Validation images and XML files directory.
VALID_DIR =  "/home/local/ASURITE/tchen169/Documents/CV4TSC/rough/RL_Model/cyclic_traffic_time_for_train/1fpsDataset/"

# Classes: 0 index is reserved for background.
# CLASSES = [
#     '__background__', 'Car', 'Motorcycle', 'Bus', 'Truck']
CLASSES = [
    '__background__', 'Vehicles']
	
# Car: Category ID 3
# Motorcycle: Category ID 4
# Airplane: Category ID 5
# Bus: Category ID 6
# Train: Category ID 7
# Truck: Category ID 8
# Boat: Category ID 9

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'