from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'yolo_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email',
    description='YOLO segmentation node publishing dynamic masks',
    license='Apache License 2.0',

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/yolo_ros2']),
        ('share/yolo_ros2', ['package.xml']),
    ],

    entry_points={
        'console_scripts': [
            'yolo_node = yolo_ros2.yolo_node:main'
        ],
    },
)

def post_install():
    script_path = os.path.join(
        'install', package_name, 'lib', package_name, 'yolo_node'
    )

    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            lines = f.readlines()

        # Replace shebang
        lines[0] = '#!/opt/venv/bin/python3\n'

        with open(script_path, 'w') as f:
            f.writelines(lines)

        print(f"[INFO] Updated shebang to use venv: {script_path}")


if __name__ == '__main__':
    setup()
    post_install()