import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='control_pkg',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[{
                'control_frequency': 50.0,
                'wheelbase': 2.8,
                'prediction_horizon': 20,
                'dt': 0.05,
                'max_steering_angle': 0.6,
                'max_steering_rate': 0.5,
                'max_acceleration': 3.0,
                'max_deceleration': -5.0,
                'max_msg_age_ms': 500.0,
                'max_lateral_error': 2.0,
                'max_heading_error': 1.0,
            }],
        ),
    ])