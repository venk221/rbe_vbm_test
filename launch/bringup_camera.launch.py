# Starts RealSense D405
#   ros2 launch ~/rbe_vbm_test/launch/bringup_camera.launch.py

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

FILTER_ARGS = ['spatial_filter.enable', 'temporal_filter.enable',
              'decimation_filter.enable', 'hole_filling_filter.enable']


def generate_launch_description():
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value='false') for name in FILTER_ARGS),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')),
            launch_arguments={'pointcloud.enable': 'true', 'align_depth.enable': 'true',
                              **{name: LaunchConfiguration(name) for name in FILTER_ARGS}}.items(),
        ),
    ])
