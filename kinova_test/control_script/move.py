import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import time

class ArmObsCollector(Node):
    def __init__(self):
        super().__init__('arm_obs_collector')
        self.bridge = CvBridge()
        self.top_img = None
        self.wrist_img = None
        self.state = None

        self.create_subscription(
            Image,
            '/world/working_living_room/model/over_table_camera/link/camera_link/sensor/camera_sensor/image',
            self.top_img_callback,
            10
        )
        self.create_subscription(
            Image,
            '/wrist_mounted_camera/image',
            self.wrist_img_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )

    def top_img_callback(self, msg):
        self.top_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def wrist_img_callback(self, msg):
        self.wrist_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def state_callback(self, msg):
        # 这里只取position，你可以根据实际需要调整
        self.state = np.array(msg.position, dtype=np.float32)

def main():
    rclpy.init()
    node = ArmObsCollector()
    client = websocket_client_policy.WebsocketClientPolicy(host="10.20.23.90", port=40003)
    task_instruction = "Move the robot arm to complete the task"
    num_steps = 50

    # 等待所有观测就绪
    print("等待图像和关节状态...")
    while node.top_img is None or node.wrist_img is None or node.state is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.1)
    print("观测已就绪，开始推理...")

    for step in range(num_steps):
        # 获取最新观测
        img = node.top_img
        wrist_img = node.wrist_img
        state = node.state

        # 预处理图像
        obs = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, 224, 224)
            ),
            "observation/state": state,
            "prompt": task_instruction,
        }

        # 推理
        action_chunk = client.infer(obs)["actions"]

        # TODO: 执行动作
        print(f"Step {step}: action_chunk = {action_chunk}")

        # 这里可以加动作执行和等待
        time.sleep(0.1)
        rclpy.spin_once(node, timeout_sec=0.01)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
