# Kinova driver + MoveIt + RViz
#   ros2 launch ~/rbe_vbm_test/launch/bringup_robot.launch.py


import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def launch_setup(context, *args, **kwargs):
    config_path = os.path.expanduser(LaunchConfiguration('config').perform(context))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    robot_ip = LaunchConfiguration('robot_ip').perform(context) or str(cfg['robot_ip'])
    go_home = LaunchConfiguration('go_home').perform(context).lower() in ('true', '1', 'yes')

    he = cfg['hand_eye']
    x, y, z = (str(v) for v in he['translation'])
    qx, qy, qz, qw = (str(v) for v in he['rotation'])

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('kinova_gen3_lite_moveit_config'), 'launch', 'robot.launch.py')),
            launch_arguments={'robot_ip': robot_ip}.items(),
        ),
        Node(
            package='tf2_ros', executable='static_transform_publisher', name='hand_eye_tf', output='screen',
            arguments=['--x', x, '--y', y, '--z', z, '--qx', qx, '--qy', qy, '--qz', qz, '--qw', qw,
                       '--frame-id', he['parent'], '--child-frame-id', he['child']],
        ),
    ]
    if go_home:
        # go_home.py 
        actions.append(TimerAction(period=8.0, actions=[ExecuteProcess(
            cmd=['python3', os.path.join(HERE, '..', 'helpers', 'go_home.py'), config_path],
            name='go_home', output='screen')]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(HERE, '..', 'config', 'views.yaml'),
                              description='Per-robot YAML with robot_ip, hand_eye, home, views, workspace'),
        DeclareLaunchArgument('robot_ip', default_value='', description='Overrides robot_ip from the YAML'),
        DeclareLaunchArgument('go_home', default_value='true', description='Move to home once MoveIt is ready'),
        OpaqueFunction(function=launch_setup),
    ])
