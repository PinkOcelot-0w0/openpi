import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionClient
import time
import math

# 关节类型，True为continuous（无限制），False为revolute（有限制）
JOINT_TYPES = [
    True,   # joint_1 (Actuator1, continuous)
    False,  # joint_2 (Actuator2, revolute)
    True,   # joint_3 (Actuator3, continuous)
    False,  # joint_4 (Actuator4, revolute)
    True,   # joint_5 (Actuator5, continuous)
    False,  # joint_6 (Actuator6, revolute)
    True,   # joint_7 (Actuator7, continuous)
]

JOINT_LIMITS = [
    (-3.14, 3.14),  # joint_1
    (-2.41, 2.41),  # joint_2
    (-3.14, 3.14),  # joint_3
    (-2.66, 2.66),  # joint_4
    (-3.14, 3.14),  # joint_5
    (-2.23, 2.23),  # joint_6
    (-3.14, 3.14),  # joint_7
]

class JointMotionTester(Node):
    def __init__(self):
        super().__init__("joint_motion_tester")
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.joint_names = [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "joint_7"
        ]

    def send_goal(self, positions):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 3
        goal_msg.trajectory.points = [point]
        self._client.wait_for_server()
        self.get_logger().info(f"Sending: {positions}")
        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.get_logger().info(f"Result: {result}")

def main():
    rclpy.init()
    node = JointMotionTester()
    for i in range(7):
        pos = [0.0] * 7
        if JOINT_TYPES[i]:
            # continuous关节，转一整圈
            pos[i] = 2 * math.pi
            node.send_goal(pos)
            time.sleep(3)
            pos[i] = 0.0
            node.send_goal(pos)
            time.sleep(3)
        else:
            # 有限制的关节，从最大转到最小
            pos[i] = JOINT_LIMITS[i][1]
            node.send_goal(pos)
            time.sleep(3)
            pos[i] = JOINT_LIMITS[i][0]
            node.send_goal(pos)
            time.sleep(3)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
