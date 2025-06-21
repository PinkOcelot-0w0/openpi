#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import time


class SimpleTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__("simple_trajectory_visualizer")

        # 简单的Marker可视化
        self.marker_pub = self.create_publisher(MarkerArray, "/simple_trajectory", 10)

        # 控制机械臂
        self.control_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )

    def visualize_and_execute(self):
        """可视化并执行圆周运动"""
        # 先发布轨迹可视化
        self.publish_trajectory_markers()

        # 执行运动
        radius = 0.5
        center1 = 0.0
        center2 = 0.5
        fixed_joints = [1.0, -1.5, 0.0, 1.0, 0.0]
        steps = 36

        for i in range(steps):
            angle = 2 * math.pi * i / steps
            joint1 = center1 + radius * math.cos(angle)
            joint2 = center2 + radius * math.sin(angle)

            msg = JointTrajectory()
            msg.joint_names = [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "joint_7",
            ]

            point = JointTrajectoryPoint()
            point.positions = [joint1, joint2] + fixed_joints
            point.time_from_start = Duration(sec=2)

            msg.points = [point]
            self.control_pub.publish(msg)

            self.get_logger().info(f"步骤 {i+1}/{steps}")
            time.sleep(2.5)

    def publish_trajectory_markers(self):
        """发布轨迹标记"""
        marker_array = MarkerArray()

        # 创建圆周轨迹点
        radius = 0.5
        center1 = 0.0
        center2 = 0.5
        steps = 36

        points = []
        for i in range(steps + 1):  # +1 to close the circle
            angle = 2 * math.pi * i / steps
            joint1 = center1 + radius * math.cos(angle)
            joint2 = center2 + radius * math.sin(angle)

            point = Point()
            point.x = joint1 * 0.5  # 缩放用于显示
            point.y = joint2 * 0.5
            point.z = 1.0
            points.append(point)

        # 轨迹线
        line_marker = Marker()
        line_marker.header.frame_id = "base_link"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "planned_trajectory"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        line_marker.scale.x = 0.02
        line_marker.color.r = 1.0
        line_marker.color.g = 0.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        line_marker.points = points
        marker_array.markers.append(line_marker)

        # 持续发布
        for _ in range(10):
            self.marker_pub.publish(marker_array)
            time.sleep(0.1)

        self.get_logger().info("轨迹标记已发布，请在RViz2中添加MarkerArray显示")


def main():
    rclpy.init()
    node = SimpleTrajectoryVisualizer()
    time.sleep(2)

    print("请在RViz2中添加MarkerArray显示，Topic: /simple_trajectory")
    input("按Enter开始...")

    node.visualize_and_execute()

    rclpy.spin(node)


if __name__ == "__main__":
    main()
