import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time


class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller")
        self.publisher = self.create_publisher(
            JointState, "/arm_controller/command", 10
        )
        self.timer = self.create_timer(0.5, self.publish_joint_positions)

        # 初始化关节位置
        self.joint_positions = [0.0] * 7  # 7个关节
        self.gripper_position = 0.0  # 夹爪位置

    def publish_joint_positions(self):
        msg = JointState()
        msg.name = [f"joint_{i+1}" for i in range(7)] + ["gripper"]
        msg.position = self.joint_positions + [self.gripper_position]
        self.publisher.publish(msg)
        self.get_logger().info(f"Published joint positions: {msg.position}")

    def set_joint_positions(self, positions):
        if len(positions) == 7:
            self.joint_positions = positions
        else:
            self.get_logger().error("Expected 7 joint positions.")

    def set_gripper_position(self, position):
        self.gripper_position = position


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()

    try:
        # 示例：设置关节位置和夹爪位置
        node.set_joint_positions([0.5, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2])  # 设置关节角度
        node.set_gripper_position(0.8)  # 设置夹爪开合程度
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
