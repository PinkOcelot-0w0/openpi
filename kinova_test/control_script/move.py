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
import cv2
import threading

"""
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7],
  points: [
    { positions: [0, 0, 0, 1.57, 0, 0.7, -1.57], time_from_start: { sec: 3 } },
  ]
}" -1
"""


class ArmObsCollector(Node):
    def __init__(self):
        super().__init__("arm_obs_collector")
        self.bridge = CvBridge()
        self.left_img = None
        # self.right_img = None
        self.wrist_img = None
        self.state = None

        self.create_subscription(
            Image,
            "/world/working_living_room/model/left_camera/link/camera_link/sensor/camera_sensor/image",
            self.left_img_callback,
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

    def left_img_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def wrist_img_callback(self, msg):
        self.wrist_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def state_callback(self, msg):
        raw_state = np.array(msg.position, dtype=np.float32)
        self.state = np.round(raw_state, 3)

    def send_goal(self, positions):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 2
        goal_msg.trajectory.points = [point]
        self._client.wait_for_server()
        # self.get_logger().info(f"目标位置: {positions}")
        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        # self.get_logger().info(f"Accepted: {result.accepted} ,Status: {result.status}")


def ros_spin(node):
    rclpy.spin(node)  # 单独处理 ROS 事件


def main():
    rclpy.init()
    node = ArmObsCollector()
    client = websocket_client_policy.WebsocketClientPolicy(
        host="10.20.23.90", port=40003
    )
    task_instruction = "抓住红色正方体"
    num_steps = 100

    print("等待图像和关节状态...")
    while node.left_img is None or node.wrist_img is None or node.state is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("观测已就绪，开始推理...")
    spin_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    spin_thread.start()
    for step in range(num_steps):
        left_img = node.left_img
        wrist_img = node.wrist_img
        state = node.state

        obs = {
            "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(left_img, 224, 224)
            ),
            "observation/wrist_image_left": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, 224, 224)
            ),
            "observation/joint_position": state[:7],
            "observation/gripper_position": (
                state[7:] if len(state) > 7 else np.array([0.0])
            ),
            "prompt": task_instruction,
        }
        print("关节位置:", state[:7])
        cv2.imshow("Left Camera", left_img)
        cv2.imshow("Wrist Camera", wrist_img)
        cv2.waitKey(1)
        action_chunk = client.infer(obs)["actions"]
        print(f"Step {step}/{num_steps}")
        for pos in action_chunk:
            safe_pos = pos[:7]
            node.send_goal(safe_pos)
    spin_thread.join()  # 等待线程结束
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
