import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import numpy as np
import time


class ArmActionClient(Node):
    def __init__(self):
        super().__init__("arm_action_client")
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
    node = ArmActionClient()
    # 用你的 action_chunk 替换下面这行
    action_chunk = np.array(
        [
            [
                -9.34891165e-02,
                7.23893647e-01,
                1.12883868e-02,
                -2.44235002e00,
                3.82699035e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                -5.35267224e-02,
                7.32235188e-01,
                1.06348400e-02,
                -2.44235002e00,
                3.43380545e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                -1.18912168e-02,
                7.46230185e-01,
                6.51635536e-03,
                -2.44235002e00,
                2.68592333e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                -5.51982840e-04,
                7.61480467e-01,
                -5.31635912e-03,
                -2.44235002e00,
                1.65655190e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                -1.20526449e-02,
                7.73464989e-01,
                -2.83574728e-02,
                -2.44235002e00,
                4.46453202e-03,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                -1.67342487e-02,
                7.79139066e-01,
                -6.32269258e-02,
                -2.44235002e00,
                -8.25919863e-03,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                -3.19621499e-03,
                7.77947278e-01,
                -1.06511453e-01,
                -2.44235002e00,
                -2.03601856e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                1.15665282e-02,
                7.71877844e-01,
                -1.51098699e-01,
                -2.44235002e00,
                -3.06538999e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                1.07798416e-02,
                7.64553124e-01,
                -1.87971714e-01,
                -2.44235002e00,
                -3.81327211e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
            [
                1.60102848e-03,
                7.59718356e-01,
                -2.08868675e-01,
                -2.44235002e00,
                -4.20645701e-02,
                -6.28968206e-01,
                -9.93321792e-01,
            ],
        ]
    )
    for i, pos in enumerate(action_chunk):
        node.get_logger().info(f"Step {i}: Moving to {pos}")
        node.send_goal(pos.tolist())
        time.sleep(2.5)  # 等待机械臂动作完成
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
