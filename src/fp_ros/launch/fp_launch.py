from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Declare the launch arguments
    mesh_file_arg = DeclareLaunchArgument(
        'mesh_file',
        default_value=os.path.join(os.getcwd(), 'demo_data/mustard0/mesh/textured_simple.obj'),
        description='Path to the mesh file'
    )

    debug_dir_arg = DeclareLaunchArgument(
        'debug_dir',
        default_value=os.path.join(os.getcwd(), 'data/debug'),
        description='Path to the debug directory'
    )

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value=os.path.join(os.getcwd(), 'data/output_frames'),
        description='Path to the output directory'
    )

    # Create the publisher node
    publisher_node = Node(
        package='fp_ros',
        executable='data_pub',
        name='data_publisher'
    )

    # Create the subscriber node
    subscriber_node = Node(
        package='fp_ros',
        executable='data_sub',
        name='data_subscriber',
        parameters=[{
            'mesh_file': LaunchConfiguration('mesh_file'),
            'debug_dir': LaunchConfiguration('debug_dir'),
            'output_dir': LaunchConfiguration('output_dir'),
            'est_refine_iter': 5,
            'track_refine_iter': 2,
            'debug': 1
        }]
    )

    # Create and return the launch description
    return LaunchDescription([
        mesh_file_arg,
        debug_dir_arg,
        output_dir_arg,
        publisher_node,
        subscriber_node
    ])