# Sawyer Docker
## Instructions
- If you are using NVIDIA graphic cards
    install nvidia-docker first.
- change directory to garage root
    ```bash
    $ cd garage_root
    ```
- build
    ```bash
    # specify gpu if you are using NVIDIA graphic cards.
    $ ./garage/contrib/ros/docker/build.sh [ robot_name ] [cpu | gpu ]
    ```
- run
    ```bash
    # cpu
    $ ./garage/contrib/ros/docker/run.sh robot_name
    # gpu
    $ ./garage/contrib/ros/docker/run_gpu.sh robot_name
    ```
- copy the intera.sh script
    The intera.sh file already exists in intera_sdk repo, copy the file into your ros workspace.

    ```bash
    $ cd /home/$USER/ros_ws/
    $ cp src/intera_sdk/intera.sh .
    ```
- customize the intera.sh script
    ```bash
    $ cd /home/$USER/ros_ws/
    $ vim intera.sh
    # Edit the 'robot_hostname' field
    # Edit the 'your_ip' field
    # Verify 'ros_version' field
    # update: ros_version='kinetic'
    ```
- start gazebo sawyer
    ```bash

    $ cd $ROS_WS
    $ ./intera.sh sim
    $ export PYTHONPATH=/opt/ros/kinetic/lib/python2.7/dist-packages:$ROS_WS/devel/lib/python3/dist-packages
    $ roslaunch sawyer_gazebo sawyer_learning.launch
    ```
- start sawyer moveit trajectory server
    ```bash
    $ cd $ROS_WS
    $ ./intera.sh sim
    $ source activate garage
    $ rosrun intera_interface enable_robot.py -e
    $ rosrun intera_interface joint_trajectory_action_server.py
    ```
- start sawyer moveit
    ```bash
    $ cd $ROS_WS
    $ ./intera.sh sim
    $ source activate garage
    # with gripper
    $ roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true
    # without gripper
    $ roslaunch sawyer_moveit_config sawyer_moveit.launch
    ```
- run launcher file
    ```bash
    $ ./intera.sh sim
    $ source activate garage
    $ python launcher_file
    ```

## Trouble Shooting
- Notice moveit sawyer collision definition.
    ```xml
    <!-- remove controller_box in sawyer_moveit/sawyer_moveit_config/srdf/sawyer.srdf.xacro -->
    <xacro:sawyer_base tip_name="$(arg tip_name)"/>
    <!--Controller Box Collisions-->
-  <xacro:if value="$(arg controller_box)">
+  <!--xacro:if value="$(arg controller_box)">
     <xacro:include filename="$(find sawyer_moveit_config)/srdf/controller_box.srdf.xacro" />
     <xacro:controller_box/>
-  </xacro:if>
+  </xacro:if-->
   <!--Right End Effector Collisions-->
   <xacro:if value="$(arg electric_gripper)">
    ```
    ```xml
    <disable_collisions link1="head" link2="right_arm_base_link" reason="Never" />
    <disable_collisions link1="head" link2="right_l0" reason="Adjacent" />
    <disable_collisions link1="head" link2="right_l1" reason="Default" />
+   <disable_collisions link1="head" link2="right_l2" reason="Default" />
    <disable_collisions link1="head" link2="screen" reason="Adjacent" />
    <disable_collisions link1="head" link2="torso" reason="Never" />
    <disable_collisions link1="pedestal" link2="right_arm_base_link" reason="Adjacent" />
    ```
