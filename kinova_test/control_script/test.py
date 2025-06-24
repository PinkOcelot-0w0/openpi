from openpi_client import image_tools
from openpi_client import websocket_client_policy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import numpy as np
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import GripperCommand
import cv2
import threading

"""
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7],
  points: [
    { positions: [0, 0, 0, 1.57, 0, 0.2, -1.57], time_from_start: { sec: 3 } },
  ]
}" -1
"""

# 在循环外初始化策略客户端。
client = websocket_client_policy.WebsocketClientPolicy(host="10.20.23.90", port=40003)

rclpy.init()
node = Node("data_collector")
bridge = CvBridge()
left_img = None
wrist_img = None
state = None
num_steps = 9999  # 本次回合要执行的步数。
task_instruction = "抓取红色正方体"  # 示例任务指令。


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


def send_trajectory(positions, time_sec=1):
    traj_msg = JointTrajectory()
    traj_msg.joint_names = [f"joint_{i+1}" for i in range(7)]
    point = JointTrajectoryPoint()
    point.positions = list(positions)  # 保证为list即可
    point.time_from_start.sec = int(time_sec)
    traj_msg.points = [point]
    traj_pub.publish(traj_msg)


def send_gripper(position: float, max_effort: float = 20.0):
    cmd = GripperCommand()
    cmd.position = float(position)
    cmd.max_effort = float(max_effort)
    gripper_pub.publish(cmd)


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
traj_pub = node.create_publisher(
    JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
)
gripper_pub = node.create_publisher(
    GripperCommand, "/robotiq_gripper_controller/gripper_cmd", 10
)

while left_img is None or wrist_img is None or state is None:
    rclpy.spin_once(node, timeout_sec=0.1)
print("数据准备完毕，开始循环控制")

cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

for step in range(num_steps):
    joint_positions = state[:7]
    gripper_position = np.array([state[7]], dtype=np.float32)
    observation = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(left_img, 224, 224)
        ),
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/joint_position": joint_positions,
        "observation/gripper_position": gripper_position,
        "prompt": task_instruction,
    }
    cv2.imshow("Left Camera", left_img)
    cv2.imshow("Wrist Camera", wrist_img)
    cv2.waitKey(1)
    print("当前关节位置", joint_positions)
    if step % 10 == 0:
        action_chunk = client.infer(observation)["actions"]
        # print(f"Step {step + 1}/{num_steps}, Action: {action_chunk}")
    next_action = np.array(action_chunk[step % 10], dtype=np.float32)
    print("下一步关节位置", step, -next_action)
    send_trajectory(next_action[:7], time_sec=0)
    send_gripper(next_action[7], max_effort=20.0)

cv2.destroyAllWindows()
