#!/bin/bash

export USER=sawyer_docker
export GARAGE_PYTHON=/opt/conda/envs/garage/bin/python

# Clean up bashrc
echo "export USER=sawyer_docker" > "/home/$USER/.bashrc"

echo "source /opt/ros/kinetic/setup.bash" >> "/home/$USER/.bashrc"
source /opt/ros/kinetic/setup.bash

echo "export USER=$USER" >> "/home/$USER/.bashrc"
echo "export HOME=/home/$USER" >> "/home/$USER/.bashrc"

export PATH=$PATH:/opt/conda/bin
echo 'export PATH=$PATH:/opt/conda/bin' >> "/home/$USER/.bashrc"

export HOME=/home/$USER

if [ ! -d "/home/$USER/ros_ws/src" ]; then
  mkdir -p "/home/$USER/ros_ws/src"
fi

# Prepare Intera SDK installation
cd "/home/$USER/ros_ws/src"
echo -e "Retrieving Instera SDK sources."
wstool init ./docker-entrypoint.sh
git clone https://github.com/RethinkRobotics/sawyer_robot.git
wstool merge sawyer_robot/sawyer_robot.rosinstall
wstool update
cd "/home/$USER/ros_ws/src"
git clone https://github.com/ros/geometry.git
cd geometry
git checkout indigo-devel
cd "/home/$USER/ros_ws/src"
git clone https://github.com/ros/geometry2.git
cd geometry2
git checkout indigo-devel

# Prepare sawyer gazebo simulator installation
cd "/home/$USER/ros_ws/src"
git clone https://github.com/RethinkRobotics/sawyer_simulator.git
wstool merge sawyer_simulator/sawyer_simulator.rosinstall
wstool update

# Prepare Moveit! installation
# Get source
cd "/home/$USER/ros_ws"
wstool merge -t src https://raw.githubusercontent.com/ros-planning/moveit/kinetic-devel/moveit.rosinstall
wstool update -t src
rosdep install -y --from-paths src --ignore-src --rosdistro kinetic
# Have to exclude some targets
# See https://github.com/ros-planning/moveit/issues/697
sed -i '/demo/s/^/# /g' "/home/$USER/ros_ws/src/moveit/moveit_ros/planning_interface/move_group_interface/CMakeLists.txt"

# Build ROS_WS
# convert python2 code to python3
chmod +x /root/code/garage/garage/contrib/ros/scripts/sawyer_2to3.sh
/root/code/garage/garage/contrib/ros/scripts/sawyer_2to3.sh "/home/$USER/ros_ws/src/"
source /opt/ros/kinetic/setup.bash
source activate garage
cd "/home/$USER/ros_ws"
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$GARAGE_PYTHON -DCATKIN_BLACKLIST_PACKAGES='moveit_setup_assistant'
