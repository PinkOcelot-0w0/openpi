from openpi_client import image_tools
from openpi_client import websocket_client_policy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import numpy as np
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import cv2
import threading
import time

"""
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7],
  points: [
    { positions: [0, 0, 0, 1.57, 0, 0.7, -1.57], time_from_start: { sec: 3 } },
  ]
}" -1

ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7],
  points: [
    { positions: [0, 0, 0, 0, 0, 0, 0], time_from_start: { sec: 3 } },
  ]
}" -1
"""

# 在循环外初始化策略客户端
client = websocket_client_policy.WebsocketClientPolicy(host="10.20.23.90", port=40003)

rclpy.init()
node = Node("data_collector")
bridge = CvBridge()
left_img = None
wrist_img = None
state = None
num_steps = 9999  # 执行的步数
task_instruction = "把黄色长方体拉出来"  # 任务


# 回调函数
def left_img_callback(msg):
    global left_img
    left_img = bridge.imgmsg_to_cv2(msg, "bgr8")


def wrist_img_callback(msg):
    global wrist_img
    wrist_img = bridge.imgmsg_to_cv2(msg, "bgr8")


def state_callback(msg):
    global state
    state = np.array(msg.position, dtype=np.float32)


vel_pub = node.create_publisher(
    JointTrajectory, "/joint_group_velocity_controller/joint_commands", 10
)


def send_velocity(velocities, time_sec=0.1):
    traj_msg = JointTrajectory()
    traj_msg.joint_names = [f"joint_{i+1}" for i in range(7)]
    point = JointTrajectoryPoint()
    point.velocities = list(velocities)
    point.time_from_start.sec = int(time_sec)
    traj_msg.points = [point]
    vel_pub.publish(traj_msg)


gripper_action_client = ActionClient(
    node, GripperCommand, "/robotiq_gripper_controller/gripper_cmd"
)


def send_gripper_action(position: float, max_effort: float = 100.0):
    from control_msgs.action import GripperCommand as GripperCommandAction

    goal_msg = GripperCommandAction.Goal()
    goal_msg.command.position = float(position)
    goal_msg.command.max_effort = float(max_effort)
    gripper_action_client.wait_for_server()
    future = gripper_action_client.send_goal_async(goal_msg)


# 订阅话题
node.create_subscription(
    Image,
    "/world/working_living_room/model/left_camera/link/camera_link/sensor/camera_sensor/image",
    left_img_callback,
    10,
)
node.create_subscription(
    Image,
    "/wrist_mounted_camera/image",
    wrist_img_callback,
    10,
)
node.create_subscription(
    JointState,
    "/joint_states",
    state_callback,
    10,
)

vel_pub = node.create_publisher(
    JointTrajectory, "/joint_group_velocity_controller/joint_commands", 10
)

# while left_img is None or wrist_img is None or state is None:
while wrist_img is None or state is None:
    rclpy.spin_once(node, timeout_sec=0)
print("数据准备完毕，开始循环控制")

threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


for step in range(num_steps):

    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(left_img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    if step % 10 == 0:
        action_chunk = client.infer(observation)["actions"]
        print(f"Step {step + 1}/{num_steps}")
        print("动作:", action_chunk)
    next_action = np.array(action_chunk[step % 10], dtype=np.float32)

    send_velocity(next_action[:7], time_sec=0)
    if next_action[7] < 0.4:
        next_action[7] = 0.05
    else:
        next_action[7] = 0.79
    send_gripper_action(next_action[7])
