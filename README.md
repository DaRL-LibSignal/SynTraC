# SynTraC: A Synthetic Dataset for Traffic Signal Control from Traffic Monitoring Cameras

SynTraC is the first public dataset designed for image-based traffic signal control. It bridges the gap between simulated environments and real-world traffic challenges by offering high-resolution images from the CARLA simulator, complete with annotations of traffic signal states and vehicle features. 

## Dataset on Hugging Face

The SynTraC dataset is now available on Hugging Face. You can download it using the following link:

[![SynTraC Dataset on Hugging Face](https://img.shields.io/badge/SynTraC-Dataset-yellow)](https://huggingface.co/datasets/Prithvi180900/SynTraC)


## Docker Support
Command to Pull the Image:
```bash
docker pull prithvi180900/my_carla_app
```

Run the Docker Image
```bash
sudo docker run --privileged --gpus all --net=host prithvi180900/my_carla_app
```
<br>



# Installation


### Installation resources
We would like to work in the docker environment which is smoothy to configure environment and reproduce problems
The whole process is [here](https://carla.readthedocs.io/en/latest/build_docker/)


#### Following this project
There some changes and bugs occurred while following the instruction.

1. This is a useful [reference](https://antc2lt.medium.com/carla-on-ubuntu-20-04-with-docker-5c2ccdfe2f71) to work with GUI
2. Enable [nvida tool-kit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3. If this happened error: <span style="color:red">XDG_RUNTIME_DIR not set in the environment.</span>, look at this [reference](https://github.com/carla-simulator/carla/issues/4755https://github.com/carla-simulator/carla/issues/4755)
```
xhost +  #disable access control
```
could directly solve it.

#### To run 

```
sudo docker run  -p 2000-2002:2000-2002  --cpuset-cpus="0-5"  --runtime=nvidia  --gpus 'all,"capabilities=graphics,utility,display,video,compute"'  -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix  -it  carlasim/carla  ./CarlaUE4.sh -opengl $1 
```

#### To config
```
docker exec -it -u root <container name> bash
```
<br>

# Setup and Evaluation Instructions


## Setup

Follow these steps to clone the repository and set up your environment.

#### Clone the repository

```bash
git clone https://github.com/prithvi1809/Syntrac.git
cd Syntrac
```

####  Install the dependencies

```bash
pip install -r requirements.txt
```

<br>

## Run the Evaluation Script
To evaluate the reinforcement learning model, run the `evaluate_rl_model.py` script. You need to specify the policy path as an argument.

### Example Command:
```bash
python evaluate_rl_model.py --policy_path cnt_dqn.pt
```

### Arguments:

- `--policy_path`: The name of the policy file located in the `policy_candidates` directory (e.g., `cnt_cql.pt`).


<br>

## Dataset Creation Script

To create a dataset for traffic simulation, you can run the `dataset_creation.py` script. This script allows you to customize the simulation parameters such as the total number of vehicles, the interval at which vehicles are spawned, and the interval at which simulation data is stored.

### Example Command:

```bash
python dataset_creation.py --total_vehicles 100 --spawn_interval 5 --store_interval 1
```

### Arguments:
- `--total_vehicles`: The total number of vehicles to spawn in the simulation. (Default: 100)
- `--spawn_interval`: The interval (in seconds) at which a new vehicle is spawned in the simulation. (Default: 5 seconds)
- `--store_interval`: The interval (in seconds) at which simulation data is captured and stored. (Default: 1 second)


<br>

## Training a Reinforcement Learning Model on a Dataset
1. Navigate to the `Train_RL_Model` folder.
2. Open the `offline_rl.ipynb` notebook.
3. Set the path to your dataset in the notebook.

<br>

## Fine-Tuning the object detection model
To download fine-tuned object detection model - [download here](https://drive.google.com/file/d/1EAaXz_svGs9_o4kChzHCUfXJpAWTfJTX/view?usp=sharing) 

For fine tuning the pretrained object detection model on your own dataset
1. Go to the `Fine_Tuning_detection_Model` directory.
2. Update the paths and other parameters in `config.py` according to your datasets and requirements.
3. Run the following command in your terminal:
    ```bash
    python train.py
    ```

For more detailed guidance, refer to this link: [Training PyTorch RetinaNet on Custom Dataset](https://debuggercafe.com/train-pytorch-retinanet-on-custom-dataset/)


<br>

<!-- # Visualization

![alt text](Images/image1.png)

![alt text](Images/image2.png) -->


<br>

# Citation
```
@article{chen2024syntrac,
  title={SynTraC: A Synthetic Dataset for Traffic Signal Control from Traffic Monitoring Cameras},
  author={Chen, Tiejin and Shirke, Prithvi and Chakravarthi, Bharatesh and Vaghela, Arpitsinh and Da, Longchao and Lu, Duo and Yang, Yezhou and Wei, Hua},
  journal={arXiv preprint arXiv:2408.09588},
  year={2024}
}
```


