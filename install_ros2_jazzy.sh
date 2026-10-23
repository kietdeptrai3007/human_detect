#!/usr/bin/env bash
# Cai ROS 2 Jazzy (RoboStack) CHI trong moi truong conda, khong cai vao he thong.
# Dung: bash install_ros2_jazzy.sh [ten_env]   (mac dinh: human_detect)
set -eo pipefail

ENV_NAME="${1:-human_detect}"

CONDA_BASE=$(conda info --base 2>/dev/null) || { echo "Khong tim thay conda"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== 1. Cai mamba (neu chua co) ==="
command -v mamba >/dev/null 2>&1 || conda install -n base -c conda-forge mamba -y

echo "=== 2. Tao/kich hoat env '$ENV_NAME' (python 3.12) ==="
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  mamba create -n "$ENV_NAME" -c conda-forge python=3.12 -y
fi
conda activate "$ENV_NAME"

echo "=== 3. Cau hinh channel RoboStack ==="
conda config --env --add channels conda-forge
conda config --env --add channels robostack-jazzy
conda config --env --set channel_priority strict

echo "=== 4. Cai ROS 2 Jazzy + phu thuoc ==="
mamba install -y \
  ros-jazzy-ros-base \
  ros-jazzy-vision-msgs \
  ros-jazzy-rqt-image-view \
  colcon-common-extensions rosdep \
  compilers cmake pkg-config ninja \
  "numpy<2" opencv librealsense

echo "=== 5. Thu vien Python cho gesture ==="
# torch / ultralytics / mediapipe: cai bang pip trong chinh env nay (xem requirements.txt)
#   pip install -r ros2_amr_UI-main/src/human_detection/requirements.txt

echo "=== Kiem tra ==="
conda deactivate
conda activate "$ENV_NAME"
ros2 --help >/dev/null && echo "ros2 OK"
python -c "import rclpy, vision_msgs, numpy; print('python OK, numpy', numpy.__version__)"
echo "XONG. Dung: conda activate $ENV_NAME, roi colcon build trong workspace."
