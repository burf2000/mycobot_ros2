import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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


class MyCobotROS2Bridge(Node):
    def __init__(self):
        super().__init__("control_sync_plan")

        # Setup device
        self.robot_m5 = os.popen("ls /dev/ttyUSB*").readline().strip()
        self.robot_wio = os.popen("ls /dev/ttyACM*").readline().strip()
        port = self.robot_m5 or self.robot_wio
        self.get_logger().info(f"Port: {port}, Baudrate: 115200")

        self.mc = MyCobot280(port, 115200)
        time.sleep(0.05)
        if self.mc.get_fresh_mode() == 0:
            self.mc.set_fresh_mode(1)
            time.sleep(0.05)

        # Joint order (must match URDF)
        self.rviz_order = [
            'link1_to_link2',
            'link2_to_link3',
            'link3_to_link4',
            'link4_to_link5',
            'link5_to_link6',
            'link6_to_link6_flange',
            'gripper_controller'
        ]

        # Subscriber to listen for target joint angles
        self.subscription = self.create_subscription(
            JointState,
            "joint_command",  # Use custom topic to avoid conflicting with MoveIt
            self.listener_callback,
            10
        )

        # Publisher for feedback joint states
        self.publisher = self.create_publisher(
            JointState,
            "joint_states",  # This is what robot_state_publisher needs
            qos_profile_sensor_data
        )

        # Timer to publish joint states at 10Hz
        self.timer = self.create_timer(0.1, self.publish_joint_states)

    def listener_callback(self, msg):
        joint_state_dict = {name: msg.position[i] for i, name in enumerate(msg.name)}

        data_list = []
        for joint in self.rviz_order:
            if joint in joint_state_dict:
                radians = joint_state_dict[joint]
                angle = round(math.degrees(radians), 3)
                data_list.append(angle)

        if len(data_list) < 7:
            return

        gripper_angle = data_list[6]
        self.mc.send_angles(data_list[:6], 35)

        if gripper_angle is not None:
            gripper_angle_clamped = max(-30, min(10, gripper_angle))
            gripper_value = int((gripper_angle_clamped + 30) * (100 / 40))
            self.mc.set_gripper_value(gripper_value, 50)

    def publish_joint_states(self):
        #print("Attempting to publish joint states...") 
        try:
            angles_deg = self.mc.get_angles()
            if angles_deg is None or len(angles_deg) != 6:
                return

            # Add gripper feedback if possible (or set dummy value)
            gripper_value = self.mc.get_gripper_value() if hasattr(self.mc, 'get_gripper_value') else 0
            angles_deg.append((gripper_value / 100) * 40 - 30)  # Invert scale

            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.rviz_order
            joint_state_msg.position = [math.radians(a) for a in angles_deg]

            self.publisher.publish(joint_state_msg)

            #self.get_logger().info(f"Published joint state: {[round(a, 2) for a in joint_state_msg.position]}")

        except Exception as e:
            self.get_logger().warn(f"Failed to read or publish joint states: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MyCobotROS2Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
