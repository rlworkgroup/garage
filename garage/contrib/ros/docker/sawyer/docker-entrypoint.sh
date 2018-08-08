#!/bin/bash

ORANGE='\033[0;33m'
WHITE='\033[1;37m'

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

export GARAGE_CODE_ROOT=/root/code/garage
echo 'export GARAGE_CODE_ROOT=/root/code/garage' >> "/home/$USER/.bashrc"

export ROS_WS=/home/$USER/ros_ws
echo 'export ROS_WS=/home/$USER/ros_ws' >> "/home/$USER/.bashrc"

export HOME=/home/$USER

if [ ! -d "/home/$USER/ros_ws/src" ]; then
  mkdir -p "/home/$USER/ros_ws/src"
fi

cd "/home/$USER/"
mkdir dev
cd dev
pushd .

# Installs a CMake project from a git repository.
# Args: rel_target_folder_name print_name repository_url
function install_from_git {
	if [ ! -d "/home/$USER/dev/$1/build" ] && [ ! -f "/home/$USER/dev/$1/build" ]; then
		echo -e "${ORANGE}Building and installing $2.${WHITE}"
		git clone $3 $1 \
			&& cd $1 \
			&& mkdir build && cd build \
			&& cmake .. && make -j4 && make install
		popd
		pushd .
	else
		echo -e "${ORANGE}$2 already exists, just need to install.${WHITE}"
		cd "/home/$USER/dev/$1/build" && make install
		popd
		pushd .
	fi
}

# Installs a CMake project from a tar.gz archive.
# Args: rel_target_folder_name print_name tar_url
function install_from_targz {
	if [ ! -d "/home/$USER/dev/$1/build" ] && [ ! -f "/home/$USER/dev/$1/build" ]; then
		echo -e "${ORANGE}Building and installing $2.${WHITE}"
		wget -O $1.tar.gz $3 \
			&& tar xzf $1.tar.gz \
			&& rm $1.tar.gz \
			&& cd $1 \
			&& mkdir build && cd build \
			&& cmake .. && make -j4 && make install
		popd
		pushd .
	else
		echo -e "${ORANGE}$2 already exists, just need to install.${WHITE}"
		cd "/home/$USER/dev/$1/build" && make install
		popd
		pushd .
	fi
}

# Installs a CMake project from a zip archive.
# Args: rel_target_folder_name print_name tar_url
function install_from_zip {
	if [ ! -d "/home/$USER/dev/$1/build" ] && [ ! -f "/home/$USER/dev/$1/build" ]; then
		echo -e "${ORANGE}Building and installing $2.${WHITE}"
		wget -O $1.zip $3 \
			&& unzip $1.zip \
			&& rm $1.zip \
			&& cd $1 \
			&& mkdir build && cd build \
			&& cmake .. && make -j4 && make install
		popd
		pushd .
	else
		echo -e "${ORANGE}$2 already exists, just need to install.${WHITE}"
		cd "/home/$USER/dev/$1/build" && make install
		popd
		pushd .
	fi
}

# Install Eigen
install_from_targz "eigen-3.3.5" "Eigen 3.3.5" "http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz"

# Install libccd
install_from_git "libccd" "libccd" "https://github.com/danfis/libccd.git"

# Install octomap
install_from_git "octomap" "Octomap" "https://github.com/OctoMap/octomap.git"

# Install fcl
if [ ! -d "/home/$USER/dev/fcl/build" ] && [ ! -f "/home/$USER/dev/fcl/build" ]; then
	echo -e "${ORANGE}Building and installing fcl.${WHITE}"
	git clone https://github.com/flexible-collision-library/fcl.git \
		&& cd fcl && git checkout fcl-0.5 \
		&& mkdir build && cd build \
		&& cmake .. && make -j4 && make install
	popd
	pushd .
else
	echo -e "${ORANGE}fcl already exists, just need to install.${WHITE}"
	cd "/home/$USER/dev/fcl/build" && make install
	popd
	pushd .
fi

# Install OMPL
install_from_targz "ompl-1.3.1" "OMPL 1.3.1" "https://github.com/ompl/ompl/archive/1.3.1.tar.gz"

# Prepare Intera SDK installation
if [ ! -d "$ROS_WS/src/sawyer_robot" ] && [ ! -f "$ROS_WS/src/sawyer_robot" ]; then
  cd "/home/$USER/ros_ws/src"
  echo -e "Retrieving Instera SDK sources."
  wstool init .
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
fi

# Prepare sawyer gazebo simulator installation
if [ ! -d "$ROS_WS/src/sawyer_simulator" ] && [ ! -f "$ROS_WS/src/sawyer_simulator" ]; then
  cd "/home/$USER/ros_ws/src"
  git clone https://github.com/RethinkRobotics/sawyer_simulator.git
  wstool merge sawyer_simulator/sawyer_simulator.rosinstall
  wstool update
fi

# Prepare sawyer moveit installation
if [ ! -d "$ROS_WS/src/sawyer_moveit" ] && [ ! -f "$ROS_WS/src/sawyer_moveit" ]; then
  cd "/home/$USER/ros_ws/src"
  wstool merge https://raw.githubusercontent.com/RethinkRobotics/sawyer_moveit/master/sawyer_moveit.rosinstall
  wstool update
fi

# Prepare Moveit! installation
# Get source
if [ ! -d "$ROS_WS/src/object_recognition_msgs" ] && [ ! -f "$ROS_WS/src/object_recognition_msgs" ]; then
  cd "/home/$USER/ros_ws/src"
  git clone https://github.com/wg-perception/object_recognition_msgs.git
fi

if [ ! -d "$ROS_WS/src/octomap_msgs" ] && [ ! -f "$ROS_WS/src/octomap_msgs" ]; then
  cd "/home/$USER/ros_ws/src"
  git clone https://github.com/OctoMap/octomap_msgs.git
  cd octomap_msgs
  git checkout indigo-devel
fi

if [ ! -d "$ROS_WS/src/urdf_parser_py" ] && [ ! -f "$ROS_WS/src/urdf_parser_py" ]; then
  cd "/home/$USER/ros_ws/src"
  git clone https://github.com/ros/urdf_parser_py.git
  cd urdf_parser_py
  git checkout indigo-devel
fi

if [ ! -d "$ROS_WS/src/warehouse_ros" ] && [ ! -f "$ROS_WS/src/warehouse_ros" ]; then
  cd "/home/$USER/ros_ws/src"
  git clone https://github.com/ros-planning/warehouse_ros.git
  cd warehouse_ros
  git checkout kinetic-devel
fi

if [ ! -d "$ROS_WS/src/rviz_visual_tools" ] && [ ! -f "$ROS_WS/src/rviz_visual_tools" ]; then
  cd "/home/$USER/ros_ws/src"
  git clone https://github.com/PickNikRobotics/rviz_visual_tools.git
  cd rviz_visual_tools
  git checkout kinetic-devel
fi

if [ ! -d "$ROS_WS/src/graph_msgs" ] && [ ! -f "$ROS_WS/src/graph_msgs" ]; then
  cd "/home/$USER/ros_ws/src"
  git clone https://github.com/davetcoleman/graph_msgs.git
  cd graph_msgs
  git checkout indigo-devel
fi

if [ ! -d "$ROS_WS/src/moveit" ] && [ ! -f "$ROS_WS/src/moveit" ]; then
 cd "/home/$USER/ros_ws"
 wstool merge -t src https://raw.githubusercontent.com/ros-planning/moveit/kinetic-devel/moveit.rosinstall
 wstool update -t src
 rosdep install -y --from-paths src --ignore-src --rosdistro kinetic
 # Have to exclude some targets
  # See https://github.com/ros-planning/moveit/issues/697
  sed -i '/demo/s/^/# /g' "/home/$USER/ros_ws/src/moveit/moveit_ros/planning_interface/move_group_interface/CMakeLists.txt"
fi

cd "/home/$USER/ros_ws"
rosdep install -y --from-paths src --ignore-src --rosdistro kinetic

# Build ROS_WS
# convert python2 code to python3
chmod +x /root/code/garage/garage/contrib/ros/scripts/sawyer_2to3.sh
/root/code/garage/garage/contrib/ros/scripts/sawyer_2to3.sh "/home/$USER/ros_ws/src/"
source /opt/ros/kinetic/setup.bash
cd "/home/$USER/ros_ws"

export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH

# See https://github.com/osmcode/pyosmium/issues/52
# We need boost_python-py35
rm /usr/lib/x86_64-linux-gnu/libboost_python.so
ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python.so

catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$GARAGE_PYTHON -DCATKIN_BLACKLIST_PACKAGES='moveit_setup_assistant'

# Copy the sawyer simulation launch file and simulation world file
cp $GARAGE_CODE_ROOT/garage/contrib/ros/envs/sawyer/sawyer_learning.launch $ROS_WS/src/sawyer_simulator/sawyer_gazebo/launch
cp $GARAGE_CODE_ROOT/garage/contrib/ros/envs/sawyer/sawyer_world_learning.launch $ROS_WS/src/sawyer_simulator/sawyer_gazebo/launch
cp $GARAGE_CODE_ROOT/garage/contrib/ros/envs/sawyer/sawyer_learning.world $ROS_WS/src/sawyer_simulator/sawyer_gazebo/worlds/

if [ -f /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so ] ; then
  rm /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so
fi

echo "
function run_gazebo() {
	QT_X11_NO_MITSHM=1 gazebo
}
" >> "/home/$USER/.bashrc"

source "/home/$USER/.bashrc"

export DBUS_SESSION_BUS_ADDRESS
export DBUS_SESSION_BUS_PID

cd "/home/$USER"

bash
