import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import time
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionClient

"""
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7],
  points: [
    { positions: [0, 0, 0, 1.57, 0, 0.7, -1.57], time_from_start: { sec: 3 } },
  ]
}" -1
"""

JOINT_LIMITS = [
    (-0.32, 6.27),  # joint_1 (continuous, 下限收紧)
    (-2.41, 2.41),  # joint_2 (revolute)
    (-6.27, 6.27),  # joint_3 (continuous)
    (-1.95, 2.57),  # joint_4 (revolute, 仿真实际极限)
    (-0.37, 6.27),  # joint_5 (continuous, 下限收紧)
    (-2.23, 0.67),  # joint_6 (revolute, 上限收紧)
    (-6.27, 6.27),  # joint_7 (continuous)
]


def clip_joints(positions):
    return [max(min(p, lim[1]), lim[0]) for p, lim in zip(positions, JOINT_LIMITS)]


class ArmObsCollector(Node):
    def __init__(self):
        super().__init__("arm_obs_collector")
        self.bridge = CvBridge()
        self.top_img = None
        self.wrist_img = None
        self.state = None

        self.create_subscription(
            Image,
            "/world/working_living_room/model/over_table_camera/link/camera_link/sensor/camera_sensor/image",
            self.top_img_callback,
            10,
        )
        self.create_subscription(
            Image, "/wrist_mounted_camera/image", self.wrist_img_callback, 10
        )
        self.create_subscription(JointState, "/joint_states", self.state_callback, 10)

        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]

    def top_img_callback(self, msg):
        self.top_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def wrist_img_callback(self, msg):
        self.wrist_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def state_callback(self, msg):
        self.state = np.array(msg.position, dtype=np.float32)

    def send_goal(self, positions):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 2
        goal_msg.trajectory.points = [point]
        self._client.wait_for_server()
        self.get_logger().info(f"Sending action goal: {positions}")
        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.get_logger().info(f"Result: {result}")


def main():
    rclpy.init()
    node = ArmObsCollector()
    client = websocket_client_policy.WebsocketClientPolicy(
        host="10.20.23.90", port=40003
    )
    task_instruction = "抓取桌上的红色方块并且放在紫色区域"
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
        print(f"Step {step}/{num_steps}")
        # 执行动作
        for pos in action_chunk:
            safe_pos = clip_joints(pos.tolist())
            node.send_goal(safe_pos)
            time.sleep(2.5)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
