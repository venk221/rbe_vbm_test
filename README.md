# Kinova Gen3 Lite - Grasping


1) In rbeadmin, open a terminal and type ssh <username>@linux.wpi.edu 
   then say yes to the certificate
2) exit and you will be back in rbeadmin
3) "LOGOUT" and restart the computer
4) log into your wpi username (hopefully it lets you log in :/)
5) git clone the code in home
6) go to interfaces, source ros, source install
7) try running this from interfaces: 
      python3 -c "from common_interfaces_merlab.srv import SendPose; print('ok')"
   hopefully it says "ok"
8) Run this - 
    grep -nE 'ROS_|RMW_|setup.bash' ~/.bashrc
    env | grep -E '^(ROS_DOMAIN_ID|ROS_AUTOMATIC_DISCOVERY_RANGE|RMW_IMPLEMENTATION)='
    ros2 pkg prefix kinova_gen3_lite_moveit_config

  You should see something like this - 
  ROS_DOMAIN_ID=42
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  /opt/ros2_kortex_ws/install/kinova_gen3_lite_moveit_config

9) cd pcl_python_merlab_build/
   rm -rf build/
   cmake -s . -B build
   cmake --build build -j4
   Run python interpreter:
        python3
        import numpy
        import pcl_python_merlab_build
        python3 -c "import numpy; print(numpy.__version__)"                # must be 1.x
        python3 -c "import pcl_python_merlab as p; print(p.pcl_version)"   # 1.14.0
      both should get imported, if not try (cp build/pcl_python_merlab*.so ~/rbe_vbm_test/)

10) Hand-eye-calibration? 

11) Terminal 1 (robot):
    cd rbe_vbm_test/ 
    ros2 launch bringup_robot.launch.py
    alternative: ros2 launch kinova_gen3_lite_moveit_config robot.launch.py robot_ip:=192.168.1.10
                 and
                 ros2 run tf2_ros static_transform_publisher --x -0.05996 --y 0.01561 --z 0.04815 --qx -0.003040 --qy -0.703660 --qz -0.015836 --qw 0.710354 --frame-id end_effector_link --child-frame-id camera_link

12) Terminal 2 (camera):
    ros2 launch realsense2_camera rs_launch.py   align_depth.enable:=true   decimation_filter.enable:=true   spatial_filter.enable:=true   temporal_filter.enable:=true   hole_filling_filter.enable:=true pointcloud.enable:=true

13) Terminal 3:
    python3 capture_views.py

14) Terminal 4:
    python3 merge_views.py
