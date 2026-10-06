"""UR3 + gripper trên fake hardware (mock_components) + MoveIt 2 + RViz
+ world_sim (thế giới + camera RGB-D ảo) + perception (OpenCV).

Chạy:  ros2 launch ur3_llm_control llm_robot.launch.py
Sau đó ở terminal khác:  ros2 run ur3_llm_control llm_commander
(hoặc thêm commander:=true để mở luôn llm_commander trong xterm)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg = get_package_share_directory('ur3_llm_control')

    moveit_config = (
        MoveItConfigsBuilder('ur3_gripper', package_name='ur3_llm_control')
        .robot_description(file_path='urdf/ur3_gripper.urdf.xacro')
        .robot_description_semantic(file_path='config/ur3_gripper.srdf')
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )

    use_rviz = LaunchConfiguration('rviz')
    scene_file = os.path.join(pkg, 'config', 'scene.yaml')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='log',
        parameters=[moveit_config.robot_description],
    )

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='log',
        parameters=[
            moveit_config.robot_description,
            os.path.join(pkg, 'config', 'ros2_controllers.yaml'),
        ],
    )

    spawners = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager'],
            output='log',
        )
        for name in ('joint_state_broadcaster', 'arm_controller', 'gripper_controller')
    ]

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict()],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='log',
        arguments=['-d', os.path.join(pkg, 'config', 'moveit.rviz')],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
        condition=IfCondition(use_rviz),
    )

    world_sim = Node(
        package='ur3_llm_control',
        executable='world_sim',
        output='screen',
        parameters=[{'scene_file': scene_file, 'seed': LaunchConfiguration('seed')}],
    )

    perception = Node(
        package='ur3_llm_control',
        executable='perception',
        output='screen',
        parameters=[{'scene_file': scene_file}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='Mở RViz'),
        DeclareLaunchArgument('seed', default_value='7', description='Seed góc xoay ban đầu của các khối'),
        world_sim,
        perception,
        robot_state_publisher,
        ros2_control_node,
        *spawners,
        move_group,
        rviz,
    ])
