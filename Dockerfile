FROM carlasim/carla:0.9.15
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
USER root

# Set the working directory
WORKDIR /app

# Copy your requirements.txt to the container
COPY requirements.txt .
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install dependencies for building Python
RUN apt-get update && apt-get install -y \
    wget \
    liblzma-dev \
    build-essential \
    libreadline-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    curl \
    libffi-dev \
    zlib1g-dev

# Download and install Python 3.8.18
RUN wget https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tgz && \
    tar xzf Python-3.8.18.tgz && \
    cd Python-3.8.18 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.8.18 Python-3.8.18.tgz

# Set Python 3.8.18 as the default python3 version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.8 1

# Verify Python version
RUN python3 --version

# Install the required Python packages
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121


# Create a non-root user
RUN useradd -ms /bin/bash carlauser

# Set the working directory
WORKDIR /app

# Set XDG_RUNTIME_DIR to avoid related warnings
ENV XDG_RUNTIME_DIR=/tmp/xdg
RUN mkdir -p /tmp/xdg && chmod 700 /tmp/xdg

# Copy your requirements.txt to the container
COPY requirements.txt .

# Install the remaining Python packages
RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

RUN python3 --version

# Copy the CARLA .whl file into the container and install it
COPY wheels/carla-0.9.15-cp38-cp38-manylinux_2_27_x86_64.whl /tmp/carla-0.9.15-cp38-cp38-manylinux_2_27_x86_64.whl
RUN pip3 install /tmp/carla-0.9.15-cp38-cp38-manylinux_2_27_x86_64.whl
USER root
# Ensure non-root user owns the necessary directories
RUN chown -R carlauser:carlauser /app /home/carla
RUN mkdir -p /run/user/$(id -u carlauser)

# Set appropriate permissions
RUN chown carlauser:carlauser /run/user/$(id -u carlauser)
RUN chmod 700 /run/user/$(id -u carlauser)
# RUN chmod -R 755 /app /home/carla
# RUN chmod +x /home/carla/CarlaUE4.sh

# Switch to the non-root user
USER carlauser
RUN export XDG_RUNTIME_DIR=/run/user/$(id -u)

# # Pre-download models to cache them in the Docker image
RUN python3 -c "from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn, ssdlite320_mobilenet_v3_large; \
                fasterrcnn_resnet50_fpn(pretrained=True); \
                maskrcnn_resnet50_fpn(pretrained=True); \
                retinanet_resnet50_fpn(pretrained=True); \
                ssdlite320_mobilenet_v3_large(pretrained=True)"


# Copy the rest of your application code to the container
COPY . .

# Switch back to root to adjust permissions before starting CARLA
USER root

# Install xdg-user-dirs if needed
RUN apt-get update && apt-get install -y xdg-user-dirs
RUN apt-get install -y xvfb
RUN apt-get install -y libx11-xcb1 libfontconfig1 libxrender1
RUN apt-get install -y libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-util1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0
RUN apt-get install -y libqt5gui5 libqt5core5a libqt5widgets5
RUN apt-get install -y x11-xserver-utils
# USER carlauser
RUN apt-get install -y screen

# USER carlauser

# Make the CarlaUE4 and the start script executable
# RUN chmod +x /home/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
RUN chmod +x /app/start_carla_and_run.sh
ENV DISPLAY=:99
# Switch back to the non-root user
USER carlauser


# Set the default command to run the start script
CMD ["/app/start_carla_and_run.sh"]
