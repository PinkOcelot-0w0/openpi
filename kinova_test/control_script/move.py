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


class ArmObsCollector(Node):
    def __init__(self):
        super().__init__("arm_obs_collector")
        self.bridge = CvBridge()
        self.left_img = None
        self.right_img = None
        self.wrist_img = None
        self.state = None

        self.create_subscription(
            Image,
            "/world/working_living_room/model/left_camera/link/camera_link/sensor/camera_sensor/image",
            self.left_img_callback,
            10,
        )
        self.create_subscription(
            Image,
            "/world/working_living_room/model/right_camera/link/camera_link/sensor/camera_sensor/image",
            self.right_img_callback,
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
        print("左侧相机图像传输完成")

    def right_img_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        print("右侧相机图像传输完成")

    def wrist_img_callback(self, msg):
        self.wrist_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        print("腕部相机图像传输完成")

    def state_callback(self, msg):
        self.state = np.array(msg.position, dtype=np.float32)
        print("关节状态传输完成")

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

    print("等待图像和关节状态...")
    while (
        node.left_img is None
        or node.right_img is None
        or node.wrist_img is None
        or node.state is None
    ):
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.1)
    print("观测已就绪，开始推理...")

    for step in range(num_steps):
        left_img = node.left_img
        right_img = node.right_img
        wrist_img = node.wrist_img
        state = node.state

        obs = {
            "observation/exterior_image_1_left": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(left_img, 224, 224)
            ),
            "observation/exterior_image_1_right": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_img, 224, 224)
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

        action_chunk = client.infer(obs)["actions"]
        print(f"Step {step}/{num_steps}")
        for pos in action_chunk:
            safe_pos = pos.tolist()
            node.send_goal(safe_pos)
            time.sleep(2.5)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
