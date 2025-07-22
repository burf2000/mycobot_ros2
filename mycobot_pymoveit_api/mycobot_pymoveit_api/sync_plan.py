import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os
import time
import math
import pymycobot
from packaging import version

# Minimum required pymycobot version
MIN_REQUIRE_VERSION = '3.6.1'

current_verison = pymycobot.__version__
print(f'Current pymycobot library version: {current_verison}')
if version.parse(current_verison) < version.parse(MIN_REQUIRE_VERSION):
    raise RuntimeError(
        f'The version of pymycobot library must be greater than {MIN_REQUIRE_VERSION}. '
        f'The current version is {current_verison}. Please upgrade the library.'
    )
else:
    print('pymycobot library version meets the requirements!')
    from pymycobot import MyCobot280

class Slider_Subscriber(Node):
    def __init__(self):
        super().__init__("control_sync_plan")
        self.subscription = self.create_subscription(
            JointState,
            "joint_states",
            self.listener_callback,
            10
        )
        self.subscription
        
        self.robot_m5 = os.popen("ls /dev/ttyUSB*").readline()[:-1]
        self.robot_wio = os.popen("ls /dev/ttyACM*").readline()[:-1]
        if self.robot_m5:
            port = self.robot_m5
        else:
            port = self.robot_wio
        self.get_logger().info(f"Port: {port}, Baudrate: 115200")
        self.mc = MyCobot280(port, 115200)
        time.sleep(0.05)
        if self.mc.get_fresh_mode() == 0:
            self.mc.set_fresh_mode(1)
            time.sleep(0.05)

        # Joint names in the desired RViz display order
        self.rviz_order = [
            'link1_to_link2',
            'link2_to_link3',
            'link3_to_link4',
            'link4_to_link5',
            'link5_to_link6',
            'link6_to_link6_flange',
            'gripper_controller'
        ]

    def listener_callback(self, msg):
        # Create a dictionary mapping joint names to their current positions
        joint_state_dict = {name: msg.position[i] for i, name in enumerate(msg.name)}

        # Reorder joint positions according to RViz display order
        data_list = []
        for joint in self.rviz_order:
            # Convert radians to degrees
            if joint in joint_state_dict:
                radians_to_angles = round(math.degrees(joint_state_dict[joint]), 3)
                data_list.append(radians_to_angles)

        print(f'data_list: {data_list}')

        gripper_angle = data_list[6]

        self.mc.send_angles(data_list[:6], 35)

        if gripper_angle is not None:
            #-30 closed, 9 max open
            # Example mapping: fully open = 0, fully closed = 100 (adjust as needed)
            # Map gripper_angle from [-30, 10] to [0, 100]
            gripper_angle_clamped = max(-30, min(10, gripper_angle))  # Clamp to valid range
            gripper_value = int((gripper_angle_clamped + 30) * (100 / 40))  # Normalize and scale    
            self.mc.set_gripper_value(gripper_value, 50)  # speed=50, tune as needed

        


def main(args=None):
    rclpy.init(args=args)
    slider_subscriber = Slider_Subscriber()
    rclpy.spin(slider_subscriber)
    slider_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
