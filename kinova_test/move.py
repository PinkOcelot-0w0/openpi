#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import math
import numpy as np

from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from cv_bridge import CvBridge
import cv2


class KinovaController(Node):
    def __init__(self):
        super().__init__("kinova_controller")
        
        self.callback_group = ReentrantCallbackGroup()
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_joint_state = None
        
        # 订阅相机图像话题
        self.image_subscription = self.create_subscription(
            Image, 
            "/wrist_mounted_camera/image", 
            self.image_callback, 
            10,
            callback_group=self.callback_group
        )
        
        # 订阅关节状态话题
        self.joint_subscription = self.create_subscription(
            JointState, 
            "/joint_states", 
            self.joint_callback, 
            10,
            callback_group=self.callback_group
        )
        
        # 尝试不同的发布方式
        # 方法1: 直接发布轨迹
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10
        )
        
        # 方法2: 使用Action Client
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # 运动参数
        self.motion_time = 0.0
        self.motion_started = False
        self.last_sent_positions = None
        self.use_action_client = False  # 先尝试直接发布
        
        # 创建定时器
        self.coordinate_timer = self.create_timer(
            1.0, 
            self.output_coordinates,
            callback_group=self.callback_group
        )
        
        # 运动更新定时器
        self.movement_timer = self.create_timer(
            5.0,  # 5秒间隔，更容易观察
            self.update_motion,
            callback_group=self.callback_group
        )
        
        # 用于线程安全的锁
        self.lock = threading.Lock()
        
        # 机械臂关节名称
        self.arm_joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 
            'joint_5', 'joint_6', 'joint_7'
        ]
        
        self.home_position = None
        self.test_step = 0
        
        self.get_logger().info("Kinova controller started - 诊断模式")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def joint_callback(self, msg):
        with self.lock:
            self.latest_joint_state = msg
            
            if self.home_position is None:
                joint_names = list(msg.name)
                joint_positions = list(msg.position)
                
                arm_positions = []
                for joint_name in self.arm_joint_names:
                    if joint_name in joint_names:
                        idx = joint_names.index(joint_name)
                        arm_positions.append(joint_positions[idx])
                
                if len(arm_positions) == 7:
                    self.home_position = arm_positions
                    self.get_logger().info(f"记录home位置: {[f'{pos:.3f}' for pos in self.home_position]}")

    def get_current_arm_positions(self):
        """获取当前机械臂关节位置"""
        with self.lock:
            if not self.latest_joint_state:
                return None
            
            joint_names = list(self.latest_joint_state.name)
            joint_positions = list(self.latest_joint_state.position)
            
            arm_positions = []
            for joint_name in self.arm_joint_names:
                if joint_name in joint_names:
                    idx = joint_names.index(joint_name)
                    arm_positions.append(joint_positions[idx])
            
            return arm_positions if len(arm_positions) == 7 else None

    def output_coordinates(self):
        """每秒输出一次机械臂的坐标"""
        try:
            current_arm_positions = self.get_current_arm_positions()
            
            if current_arm_positions:
                print("=" * 70)
                print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 测试步骤: {self.test_step}")
                print("当前机械臂关节位置:", [f'{pos:.4f}' for pos in current_arm_positions])
                
                if self.last_sent_positions:
                    print("上次发送目标位置:", [f'{pos:.4f}' for pos in self.last_sent_positions])
                    diff = [abs(current - target) for current, target in zip(current_arm_positions, self.last_sent_positions)]
                    print("位置差异:", [f'{d:.4f}' for d in diff])
                    max_diff = max(diff)
                    
                    if max_diff > 0.001:  # 1毫弧度的差异
                        print(f"状态: ❌ 机械臂未响应 (最大差异: {max_diff:.4f})")
                        if self.test_step > 2:
                            print("🔧 建议检查:")
                            print("   1. Gazebo仿真是否暂停？")
                            print("   2. 机械臂是否在安全模式？")
                            print("   3. 控制器配置是否正确？")
                    else:
                        print(f"状态: ✅ 机械臂正在移动")
                
                print(f"轨迹发布器订阅者数量: {self.joint_trajectory_pub.get_subscription_count()}")
                print(f"Action服务器可用: {self.action_client.server_is_ready()}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to get coordinates: {e}")

    def create_safe_trajectory(self, target_positions):
        """创建安全的轨迹消息"""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"  # 尝试设置frame_id
        
        msg.joint_names = self.arm_joint_names.copy()
        
        # 创建多个点的轨迹，确保平滑运动
        points = []
        
        # 起始点（当前位置）
        current_pos = self.get_current_arm_positions()
        if current_pos:
            start_point = JointTrajectoryPoint()
            start_point.positions = current_pos
            start_point.velocities = [0.0] * 7
            start_point.accelerations = [0.0] * 7
            start_point.time_from_start.sec = 0
            start_point.time_from_start.nanosec = 0
            points.append(start_point)
        
        # 目标点
        end_point = JointTrajectoryPoint()
        end_point.positions = target_positions
        end_point.velocities = [0.0] * 7
        end_point.accelerations = [0.0] * 7
        end_point.time_from_start.sec = 5  # 5秒执行时间
        end_point.time_from_start.nanosec = 0
        points.append(end_point)
        
        msg.points = points
        return msg

    def send_trajectory_via_action(self, target_positions):
        """通过Action Client发送轨迹"""
        if not self.action_client.server_is_ready():
            self.get_logger().warn("Action服务器未准备好")
            return
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = self.create_safe_trajectory(target_positions)
        
        self.get_logger().info("通过Action Client发送轨迹...")
        
        future = self.action_client.send_goal_async(goal_msg)
        
        def goal_response_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("轨迹被拒绝!")
            else:
                self.get_logger().info("轨迹被接受!")
        
        future.add_done_callback(goal_response_callback)

    def test_different_approaches(self):
        """测试不同的控制方法"""
        if not self.home_position:
            self.get_logger().warn("等待home位置...")
            return
        
        self.test_step += 1
        
        # 生成一个小的测试移动
        test_positions = self.home_position.copy()
        
        if self.test_step == 1:
            # 测试1: 单关节小幅移动
            test_positions[0] += 0.05
            self.get_logger().info("🧪 测试1: 单关节小幅移动 (joint_1 +0.05)")
            
        elif self.test_step == 2:
            # 测试2: 反向移动
            test_positions[0] -= 0.05
            self.get_logger().info("🧪 测试2: 反向移动 (joint_1 -0.05)")
            
        elif self.test_step == 3:
            # 测试3: 两个关节同时移动
            test_positions[0] += 0.03
            test_positions[1] += 0.03
            self.get_logger().info("🧪 测试3: 两关节移动 (joint_1,2 +0.03)")
            
        elif self.test_step == 4:
            # 测试4: 回到home位置
            test_positions = self.home_position.copy()
            self.get_logger().info("🧪 测试4: 回到home位置")
            
        elif self.test_step >= 5:
            # 开始圆周运动
            radius = 0.02
            angle = (self.test_step - 5) * 0.3
            test_positions[0] = self.home_position[0] + radius * math.cos(angle)
            test_positions[1] = self.home_position[1] + radius * math.sin(angle)
            self.get_logger().info(f"🎯 圆周运动: 角度={angle:.2f}")
        
        self.last_sent_positions = test_positions.copy()
        
        # 尝试两种发送方式
        if self.use_action_client and self.test_step > 2:
            self.send_trajectory_via_action(test_positions)
        else:
            # 直接发布
            traj_msg = self.create_safe_trajectory(test_positions)
            self.joint_trajectory_pub.publish(traj_msg)
            
            if self.test_step == 2:
                self.get_logger().info("下次将尝试Action Client...")
                self.use_action_client = True

    def update_motion(self):
        """更新运动"""
        try:
            self.test_different_approaches()
        except Exception as e:
            self.get_logger().error(f"Motion update failed: {e}")

    def display_camera_feed(self):
        """实时显示相机内容"""
        while rclpy.ok():
            try:
                with self.lock:
                    if self.latest_image is not None:
                        display_image = self.latest_image.copy()
                        
                        timestamp = time.strftime('%H:%M:%S')
                        cv2.putText(display_image, f"Time: {timestamp}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        cv2.putText(display_image, f"Test Step: {self.test_step}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        method = "Action Client" if self.use_action_client else "Direct Publish"
                        cv2.putText(display_image, f"Method: {method}", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        cv2.imshow("Kinova Wrist Camera", display_image)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                    else:
                        waiting_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(waiting_image, "Waiting for camera feed...", 
                                  (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.imshow("Kinova Wrist Camera", waiting_image)
                        cv2.waitKey(100)
                        
            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(0.1)

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = KinovaController()
        
        executor = MultiThreadedExecutor()
        executor.add_node(controller)
        
        camera_thread = threading.Thread(target=controller.display_camera_feed, daemon=True)
        camera_thread.start()
        
        print("=" * 60)
        print("Kinova诊断控制器")
        print("- 测试不同的控制方法")
        print("- 逐步增加运动复杂度")
        print("- 监控机械臂响应")
        print("按 Ctrl+C 停止程序")
        print("=" * 60)
        
        time.sleep(3)
        
        # 额外诊断信息
        print("\n🔍 系统诊断:")
        print("请在另一个终端检查以下内容:")
        print("1. Gazebo是否在运行且未暂停?")
        print("   - 在Gazebo界面点击播放按钮")
        print("2. 检查控制器状态:")
        print("   ros2 control list_controllers")
        print("3. 检查硬件接口:")
        print("   ros2 control list_hardware_components")
        print("4. 监控轨迹话题:")
        print("   ros2 topic hz /joint_trajectory_controller/joint_trajectory")
        print()
        
        executor.spin()
        
    except KeyboardInterrupt:
        print("\n正在停止程序...")
    finally:
        cv2.destroyAll_windows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()