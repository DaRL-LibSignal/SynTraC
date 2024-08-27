#!/bin/bash

# Start Xvfb in the background
cd /home/carla
Xvfb :99 -screen 0 1920x1080x24 &
sleep 5  # Wait for Xvfb to initialize

# Start Carla simulator in a detached screen session
nohup xvfb-run -a --server-args='-screen 0 1920x1080x24' ./CarlaUE4.sh -RenderOffScreen > carla_output.log 2>&1 &

# Wait for a few seconds to ensure Carla has started
sleep 10

# Run your Python script
cd /app
python3 -u evaluate_rl_model.py

# Keep the container running
tail -f /dev/null
