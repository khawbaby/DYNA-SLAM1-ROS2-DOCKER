# ===============================================================================
# Default to cpu build. CUDA kernels are built only if `--target nvidia_gpu`
# ===============================================================================
ARG TARGET=cpu

# ===============================================================================
# Base stage (Common dependencies for CPU and NVIDIA GPU builds)
# ===============================================================================
FROM osrf/ros:humble-desktop-full-jammy AS base
ARG USE_CI

RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive

# Basic system packages
RUN apt-get install -y \
    gnupg2 \
    curl \
    lsb-core \
    vim \
    wget \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    unzip \
    pkg-config \
    libpng16-16 \
    libjpeg-turbo8 \
    libtiff5 \
    libeigen3-dev \
    libglew-dev \
    libgl1-mesa-dev \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    nano \
    tmux

# ROS packages
# Install ROS + dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-pcl-ros \
    ros-humble-nav2-common \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-common \
    ros-humble-vision-opencv \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-topic-tools \
    ros-humble-tf-transformations \
    ros-humble-rviz2 \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-image-transport-plugins \
    python3-colcon-common-extensions \
    usbutils \
    udev \
    libgtk-3-0 \
    libglew2.2 \
    libgl1-mesa-glx \
    x11-apps \
    gdb \
    gdbserver 

# ------------------------------------------------------------------------------
# OpenCV build dependencies
# ------------------------------------------------------------------------------
RUN apt-get install -y \
    python2-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev

# ------------------------------------------------------------------------------
# Build OpenCV 4.4
# ------------------------------------------------------------------------------
RUN cd /tmp && git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.4.0 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_TESTS=OFF \
          -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j8 && \
    make install && \
    cd / && rm -rf /tmp/opencv

# ------------------------------------------------------------------------------
# Python Environment Fixes
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Python Virtual Environment
# ------------------------------------------------------------------------------

RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# Upgrade pip inside venv
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA) inside venv
RUN /opt/venv/bin/pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install YOLO inside venv
RUN /opt/venv/bin/pip install ultralytics

# Force compatible versions (still inside venv)
RUN /opt/venv/bin/pip install numpy==1.26.4 opencv-python==4.5.5.64 --force-reinstall

# Optional acceleration
RUN /opt/venv/bin/pip install onnxruntime-gpu

# ------------------------------------------------------------------------------
# VSCode install script (optional dev tool)
# ------------------------------------------------------------------------------
COPY ./container_root/shell_scripts/vscode_install.sh /root/
RUN cd /root/ && chmod +x * && ./vscode_install.sh && rm -rf vscode_install.sh


# ===============================================================================
# NVIDIA GPU image stage
# ===============================================================================
FROM nvidia/opengl:1.0-glvnd-devel-ubuntu18.04 AS glvnd
FROM base AS nvidia_gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2

COPY --from=glvnd /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
                  /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:-all}

# ------------------------------------------------------------------------------
# CUDA toolkit
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    gnupg

RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && rm -rf cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-2

ENV PATH=/usr/local/cuda-12.2/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}

# ------------------------------------------------------------------------------
# Pangolin
# ------------------------------------------------------------------------------
COPY FastTrack/Thirdparty/Pangolin /tmp/Pangolin

RUN cd /tmp/Pangolin && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS=-std=c++14 \
          -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j8 && \
    make install && \
    cd / && rm -rf /tmp/Pangolin && ldconfig

# ------------------------------------------------------------------------------
# ORB-SLAM3 + ROS wrapper
# ------------------------------------------------------------------------------
COPY FastTrack /home/orb/ORB_SLAM3
COPY orb_slam3_ros2_wrapper /root/colcon_ws/src/orb_slam3_ros2_wrapper
COPY orb_slam3_map_generator /root/colcon_ws/src/orb_slam3_map_generator
COPY slam_msgs /root/colcon_ws/src/slam_msgs

RUN if [ "$USE_CI" = "true" ]; then \
    . /opt/ros/humble/setup.sh && \
    cd /home/orb/ORB_SLAM3 && mkdir -p build && ./build.sh && \
    . /opt/ros/humble/setup.sh && \
    cd /root/colcon_ws && colcon build --symlink-install; \
    fi

RUN rm -rf /home/orb/ORB_SLAM3 /root/colcon_ws


# ===============================================================================
# CPU stage
# ===============================================================================
FROM base AS cpu

# Build Pangolin
RUN cd /tmp && git clone https://github.com/stevenlovegrove/Pangolin && \
    cd Pangolin && git checkout v0.9.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS=-std=c++14 \
          -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j8 && \
    make install && \
    cd / && rm -rf /tmp/Pangolin && ldconfig

COPY ORB_SLAM3 /home/orb/ORB_SLAM3
COPY orb_slam3_ros2_wrapper /root/colcon_ws/src/orb_slam3_ros2_wrapper
COPY orb_slam3_map_generator /root/colcon_ws/src/orb_slam3_map_generator
COPY slam_msgs /root/colcon_ws/src/slam_msgs

RUN if [ "$USE_CI" = "true" ]; then \
    . /opt/ros/humble/setup.sh && \
    cd /home/orb/ORB_SLAM3 && mkdir -p build && ./build.sh && \
    . /opt/ros/humble/setup.sh && \
    cd /root/colcon_ws && colcon build --symlink-install; \
    fi

RUN rm -rf /home/orb/ORB_SLAM3 /root/colcon_ws


# ===============================================================================
# Final stage
# ===============================================================================
FROM ${TARGET} AS final