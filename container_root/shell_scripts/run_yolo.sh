#!/bin/bash
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash
source /opt/venv/bin/activate

sed -i '1s|.*|#!/opt/venv/bin/python3|' \
/root/colcon_ws/install/yolo_ros2/lib/yolo_ros2/yolo_node

/root/colcon_ws/install/yolo_ros2/lib/yolo_ros2/yolo_node
