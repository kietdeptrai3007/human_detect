import os
from glob import glob

from setuptools import setup

package_name = 'human_detection'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, package_name + '.gesture', package_name + '.tracking'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt') + glob('models/*.task')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kiet',
    maintainer_email='nguyenvankiet3007@gmail.com',
    description='ROS 2 node phat hien nguoi (YOLO) va nhan dien cu chi (DD-Net) tren /image_front',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'gesture_node = human_detection.gesture_node:main',
            'video_publisher = human_detection.video_publisher:main',
            'trajectory_node = human_detection.trajectory_node:main',
            'record_node = human_detection.record_node:main',
        ],
    },
)
