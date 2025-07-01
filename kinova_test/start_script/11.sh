source ~/data/ros2_kortex_ws/install/setup.bash
#ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 gripper:=robotiq_2f_85 dof:=7
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10 gripper:=robotiq_2f_85 use_internal_bus_gripper_comm:=true launch_rviz:=true dof:=7