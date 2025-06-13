#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__("camera_subscriber")
        self.bridge = CvBridge()
        self.latest_image = None  # 保存最新图像
        self.latest_joint_state = None  # 保存最新关节状态

        # 订阅相机图像话题
        self.subscription = self.create_subscription(
            Image, "/wrist_mounted_camera/image", self.image_callback, 10
        )
        # 订阅关节状态话题
        self.joint_subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10
        )

        self.get_logger().info("Camera subscriber node started.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv_image
            # 不再输出相机信息
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def joint_callback(self, msg):
        self.latest_joint_state = msg
        # 只在终端输出关节位置
        print("当前机械臂关节位置:", msg.position)


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    try:
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
        num_steps = 4  # 步数

        for step in range(num_steps):
            rclpy.spin_once(node, timeout_sec=0.1)  # 获取最新消息
            img = node.latest_image
            wrist_img = node.latest_image
            # 获取机械臂关节状态
            if node.latest_joint_state is not None:
                state = list(node.latest_joint_state.position)
            else:
                print("等待机械臂关节状态...")
                continue
            task_instruction = "让机械臂绕圈"  # 你可以根据需要修改

            observation = {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, 224, 224)
                ),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, 224, 224)
                ),
                "observation/state": state,
                "prompt": task_instruction,
            }

            action_chunk = client.infer(observation)["actions"]
            # 在这里执行 action_chunk
            print(f"Step {step}, Action: {action_chunk}")

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
