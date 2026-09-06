from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'hunav_isaac_wrapper'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        # Install package.xml
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        
        # Install config files
        (os.path.join('share', package_name, 'config'),
            glob('config/**/*', recursive=True)),
        
        # Install scenario files
        (os.path.join('share', package_name, 'scenarios'),
            glob('scenarios/*.yaml')),
        
        # Install map files
        (os.path.join('share', package_name, 'maps'),
            glob('maps/*')),
        
        # Install world files (top-level USDs + nested assets/, e.g. museum meshes)
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.usd') + glob('worlds/*.usd-cache/**/*', recursive=True)),
        (os.path.join('share', package_name, 'worlds', 'assets', 'museum'),
            [p for p in glob('worlds/assets/museum/*') if not os.path.isdir(p)]),
        (os.path.join('share', package_name, 'worlds', 'assets', 'museum', 'textures'),
            glob('worlds/assets/museum/textures/*')),
        (os.path.join('share', package_name, 'worlds', 'assets', 'hospital'),
            [p for p in glob('worlds/assets/hospital/*') if not os.path.isdir(p)]),
        (os.path.join('share', package_name, 'worlds', 'assets', 'hospital', 'textures'),
            glob('worlds/assets/hospital/textures/*')),
        (os.path.join('share', package_name, 'worlds', 'assets', 'office'),
            [p for p in glob('worlds/assets/office/*') if not os.path.isdir(p)]),
        (os.path.join('share', package_name, 'worlds', 'assets', 'office', 'textures'),
            glob('worlds/assets/office/textures/*')),
        (os.path.join('share', package_name, 'worlds', 'assets', 'bookstore'),
            [p for p in glob('worlds/assets/bookstore/*') if not os.path.isdir(p)]),
        (os.path.join('share', package_name, 'worlds', 'assets', 'bookstore', 'textures'),
            glob('worlds/assets/bookstore/textures/*')),
        (os.path.join('share', package_name, 'worlds', 'assets', 'house_museum'),
            [p for p in glob('worlds/assets/house_museum/*') if not os.path.isdir(p)]),
        (os.path.join('share', package_name, 'worlds', 'assets', 'house_museum', 'textures'),
            glob('worlds/assets/house_museum/textures/*')),
        (os.path.join('share', package_name, 'worlds', 'assets', 'small_house'),
            [p for p in glob('worlds/assets/small_house/*') if not os.path.isdir(p)]),
        (os.path.join('share', package_name, 'worlds', 'assets', 'small_house', 'textures'),
            glob('worlds/assets/small_house/textures/*')),
        (os.path.join('share', package_name, 'worlds', 'assets', 'small_warehouse'),
            [p for p in glob('worlds/assets/small_warehouse/*') if not os.path.isdir(p)]),
        (os.path.join('share', package_name, 'worlds', 'assets', 'small_warehouse', 'textures'),
            glob('worlds/assets/small_warehouse/textures/*')),
        
        # Install behavior tree files
        (os.path.join('share', package_name, 'behavior_trees'),
            glob('behavior_trees/*.xml')),
        
        # Install other files
        (os.path.join('share', package_name),
            ['README.md', 'isaacsim.exp.base.kit']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'geometry_msgs',
        'std_msgs',
        'nav_msgs',
        'sensor_msgs',
        'tf2_ros',
        'tf2_geometry_msgs',
        'hunav_msgs',
        'pyyaml',
        'numpy',
        'matplotlib',
    ],
    zip_safe=True,
    maintainer='Miguel Escudero Jiménez',
    maintainer_email='mescjim@upo.es',
    description='Isaac Sim wrapper for HuNavSim human navigation simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hunav_isaac_main = scripts.main:main',
            'hunav_isaac_launcher = hunav_isaac_wrapper.ros_launcher:main',
        ],
        'ros2pkg': [
            'hunav_isaac_wrapper = hunav_isaac_wrapper'
        ],
    },
)
