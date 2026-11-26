"""Gambit in Gazebo: world, arm, controllers, bridges and (optionally) the gambit_ros nodes.

Reproduces the manual steps in docs/gazebo.md:
  1. gz sim on gambit_bringup/world/chess_world.sdf with GZ_SIM_RESOURCE_PATH set
  2. robot_state_publisher with the xacro-generated robot_description
  3. ros_gz_sim create (spawn the arm), then the controller spawners
  4. ros_gz_bridge with gz_bridge.yaml, plus the set_pose service bridge
  5. optional: vision_node, game_node, arm_node from gambit_ros (start_nodes:=true)

Usage (after colcon build + source install/setup.bash):
  ros2 launch gambit_bringup gambit.launch.py
  ros2 launch gambit_bringup gambit.launch.py headless:=true start_nodes:=true

This is the one Python file kept as .py on this branch: ros2 launch needs it.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

WORLD_NAME = 'gambit_chess'


def _gz_sim(context, world_file):
    """Include ros_gz_sim's gz_sim.launch.py with -r (and -s --headless-rendering if headless)."""
    headless = LaunchConfiguration('headless').perform(context).lower() in ('true', '1', 'yes')
    gz_args = f"-r {'-s --headless-rendering ' if headless else ''}{world_file}"
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items())]


def generate_launch_description():
    bringup = get_package_share_directory('gambit_bringup')
    description = get_package_share_directory('gambit_description')

    world_file = os.path.join(bringup, 'world', 'chess_world.sdf')
    controllers_file = os.path.join(bringup, 'config', 'controllers.yaml')
    bridge_file = os.path.join(bringup, 'config', 'gz_bridge.yaml')
    xacro_file = os.path.join(description, 'urdf', 'so101.urdf.xacro')

    # Models (board, camera, pieces) and package:// meshes. package://gambit_description/... is
    # resolved by Gazebo relative to the folder that contains the gambit_description share dir.
    resource_path = os.pathsep.join(filter(None, [
        os.path.join(bringup, 'models'),
        os.path.join(bringup, 'models', 'pieces'),
        os.path.dirname(description),
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ]))

    use_sim_time = LaunchConfiguration('use_sim_time')
    start_nodes = LaunchConfiguration('start_nodes')

    # robot_description from xacro, passed as a string parameter (a params value, not a CLI arg:
    # see the gotcha in docs/gazebo.md about -p robot_description crashing).
    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file,
                 ' use_sim:=true', ' controllers_file:=', controllers_file]),
        value_type=str)

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': use_sim_time}])

    spawn_arm = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-world', WORLD_NAME, '-topic', 'robot_description', '-name', 'gambit_arm'])

    controller_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=['joint_state_broadcaster', 'arm_controller', 'gripper_controller',
                   '--controller-manager', '/controller_manager'])

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge', output='screen',
        parameters=[{'config_file': bridge_file, 'use_sim_time': use_sim_time}])

    set_pose_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='set_pose_bridge', output='screen',
        arguments=[f'/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose'])

    # Optional gambit_ros nodes (C++). Their params file is looked up lazily so the sim still
    # launches when gambit_ros is not built.
    ros_params = PathJoinSubstitution([LaunchConfiguration('gambit_ros_share'), 'config', 'params.yaml'])
    gambit_nodes = [
        Node(package='gambit_ros', executable=exe, output='screen', condition=IfCondition(start_nodes),
             parameters=[ros_params, {'use_sim_time': use_sim_time}])
        for exe in ('vision_node', 'game_node', 'arm_node')
    ]
    try:
        default_ros_share = get_package_share_directory('gambit_ros')
    except Exception:   # PackageNotFoundError: gambit_ros not built
        default_ros_share = ''

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run gz sim server only with headless rendering'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('start_nodes', default_value='false',
                              description='Also start gambit_ros vision/game/arm nodes'),
        DeclareLaunchArgument('gambit_ros_share', default_value=default_ros_share,
                              description='Share dir of gambit_ros (for config/params.yaml)'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        OpaqueFunction(function=_gz_sim, args=[world_file]),
        robot_state_publisher,
        bridge,
        set_pose_bridge,
        spawn_arm,
        # Controllers can only load once the arm (and its gz_ros2_control plugin) exists.
        RegisterEventHandler(OnProcessExit(target_action=spawn_arm, on_exit=[controller_spawner])),
        *gambit_nodes,
    ])
