## Support Platform
| Robot        | Workstation OS | ROS    | Python | Simulation Platform |
|:------------:|:--------------:| :-----:|:------:|:-------------------:|
| Sawyer       | Ubuntu16.04    | Kinetic|   3    |Gazebo 7.0           |
### Major python packages dependencies
- catkin_pkg
- rospkg
- defusedxml
- opencv-python
## Define bash variables which will be used throughout the tutorial
### Define garage conda env python interpreter, garage path, your favorite workspace name as bash variable
```bash
$ source activate garage
$ export GARAGE_PYTHON=`which python`
$ source deactivate
$ export ROS_WS=/path/to/your/ros/workspace/
$ export GARAGE=/path/to/your/garage/
```
## Installation
- Assuming we got a clean system which only has garage running.
- Check if our robots and workstations are supported by garage.contrib.ros
- Following is how we setup the garage.contrib.ros's environment for sawyer robot
### Setup Workstation
#### INSTALL ROS
##### Configure Ubuntu repositories
Configure your Ubuntu repositories to allow "restricted," "universe," and "multiverse." **Likely, they are already configured properly, and you only need to confirm the configuration.**
##### Setup your sources.list
    $ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
##### Setup your keys
    $ sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
##### Update to Latest Software Lists
    $ sudo apt-get update
##### Install ROS Kinetic Desktop Full
    $ sudo apt-get install ros-kinetic-desktop-full
##### Initialize rosdep
rosdep enables you to easily install system dependencies for source you want to compile and is required to run some core components in ROS.
```bash
$ sudo rosdep init
$ rosdep update
```
##### Install rosinstall
    $ sudo apt-get install python-rosinstall
#### CREATE DEVELOPMENT WORKSPACE
##### Install Python Dependecies
```bash
$ source activate garage
$ pip install catkin_pkg
$ pip install rospkg
$ pip install defusedxml
$ pip install opencv-python
```
##### Create ROS Workspace
    $ mkdir -p $ROS_WS/src
##### Source ROS Setup
    $ source /opt/ros/kinetic/setup.bash
You can also add this to your .bashrc by using

    $ echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
##### Build
```bash
$ cd $ROS_WS
$ catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$GARAGE_PYTHON
```
#### INSTALL INTERA SDK DEPENDENCIES
##### Install SDK Dependencies
```bash
$ sudo apt-get update
$ sudo apt-get install git-core python-argparse python-wstool python-vcstools python-rosdep ros-kinetic-control-msgs ros-kinetic-joystick-drivers ros-kinetic-xacro ros-kinetic-tf2-ros ros-kinetic-rviz ros-kinetic-cv-bridge ros-kinetic-actionlib ros-kinetic-actionlib-msgs ros-kinetic-dynamic-reconfigure ros-kinetic-trajectory-msgs ros-kinetic-rospy-message-converter
```
#### INSTALL INTERA ROBOT SDK
##### Activate the conda environment for garage
    $ source activate garage
**garage is your garage conda environment's name**
##### Download the SDK on your Workstation
```bash
$ cd $ROS_WS/src
$ wstool init .
$ git clone https://github.com/RethinkRobotics/sawyer_robot.git
$ wstool merge sawyer_robot/sawyer_robot.rosinstall
$ wstool update
```
##### Download the dependencies
```bash
$ cd $ROS_WS/src
$ git clone https://github.com/ros/geometry.git
$ cd geometry
$ git checkout indigo-devel
$ cd ..
$ git clone https://github.com/ros/geometry2.git
$ cd geometry2
$ git checkout indigo-devel
```
##### Source ROS Setup
    $ source /opt/ros/kinetic/setup.bash
##### Build
```bash
$ cd $ROS_WS
$ catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$GARAGE_PYTHON
```
#### GONFIGURE ROBOT COMMUNICATION/ROS WORKSPACE
##### intera.sh ROS Environment Setup
##### Copy the intera.sh script
The intera.sh file already exists in intera_sdk repo, copy the file into your ros workspace.

    $ cp $ROS_WS/src/intera_sdk/intera.sh ~/$ROS_WS
##### Customize the intera.sh script
```bash
$ cd $ROS_WS
$ gedit intera.sh
```
##### Edit the 'robot_hostname' field
##### Edit the 'your_ip' field
##### Verify 'ros_version' field
update: ros_version='kinetic'
##### Save and Close intera.sh script
### Setup Gazebo Simulation for Sawyer
**Please skip this section if you don't want simulation**
#### Installation/Prerequisites
- Make sure you have finished [workstation setup](#setup-workstation)
- Ensure the following software packages are installed:
```bash
$ sudo apt-get install gazebo7 ros-kinetic-qt-build ros-kinetic-gazebo-ros-control ros-kinetic-gazebo-ros-pkgs ros-kinetic-ros-control ros-kinetic-control-toolbox ros-kinetic-realtime-tools ros-kinetic-ros-controllers ros-kinetic-xacro python-wstool ros-kinetic-tf-conversions ros-kinetic-kdl-parser ros-kinetic-sns-ik-lib
```
#### Sawyer Simulator Installation
##### Install sawyer_simulator
```bash
$ cd $ROS_WS/src
$ git clone https://github.com/RethinkRobotics/sawyer_simulator.git
$ wstool merge sawyer_simulator/sawyer_simulator.rosinstall
$ wstool update
```
##### Build Source
```bash
$ source /opt/ros/kinetic/setup.bash
$ cd $ROS_WS
$ catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$GARAGE_PYTHON
```
### Setup MoveIt! for safety check
#### Install MoveIt!
```bash
$ cd $ROS_WS
$ wstool merge -t src https://raw.githubusercontent.com/ros-planning/moveit/kinetic-devel/moveit.rosinstall
$ wstool update -t src
$ rosdep install -y --from-paths src --ignore-src --rosdistro kinetic
$ catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$GARAGE_PYTHON
```
### Convert Python2 code in intera packages to Python3
- As our environments only support Python3, we need to convert Python2 code in intera packages.
- These can be done using garage.contrib.ros.scripts.sawyer_2to3.sh
##### Customize sawyer_2to3.sh
Upgrade path:

    ros_ws=$ROS_WS/src/
Remove sawyer_simulator, if you don't need it:
##### Execute script
```bash
$ ./$GARAGE/contrib/ros/scripts/sawyer_2to3.sh
```
## Usage
### Use Gazebo simulation
- This section is only for simulation user
#### Copy the sawyer simulation launch file and simulation world file
```bash
$ cp $GARAGE/contrib/ros/envs/sawyer/sawyer_learning.launch $ROS_WS/src/sawyer_simulator/sawyer_gazebo/launch
$ cp $GARAGE/contrib/ros/envs/sawyer/sawyer_world_learning.launch $ROS_WS/src/sawyer_simulator/sawyer_gazebo/launch
$ cp $GARAGE/contrib/ros/envs/sawyer/sawyer_learning.world $ROS_WS/src/sawyer_simulator/sawyer_gazebo/worlds/
```
#### Copy the model file
```bash
$ cp $GARAGE/contrib/ros/envs/sawyer/models/target $ROS_WS/src/sawyer_simulator/sawyer_sim_examples/models
```
#### Set the environment variables
**Make sure the garage conda env is not activated now!**
```bash
$ cd $ROS_WS
$ SHELL=/bin/bash ./intera.sh sim
$ export PYTHONPATH=/opt/ros/kinetic/lib/python2.7/dist-packages:$ROS_WS/devel/lib/python3/dist-packages
```
#### Launch the gazebo sawyer
    $ roslaunch sawyer_gazebo sawyer_learning.launch
### Use real Sawyer
#### If you are using vicon system
Please append your vicon ros topics in your config_personal file:
    ```
    # e.g.
    VICON_TOPICS = ['vicon/vicon_object/cube']
    ```
### Launch MoveIt! sawyer for safety check
Follow this [intera official tutorial](http://sdk.rethinkrobotics.com/intera/MoveIt_Tutorial)
### Run the training script
#### Open a new terminator
#### Set the environment variables
```bash
$ cd $ROS_WS
$ SHELL=/bin/bash ./intera.sh sim
$ source activate garage
```
#### Configure experiment
**Remember to add STEP_FREQ in your personal config_personal file!**
    ```
    # e.g.
    STEP_FREQ = 20
    ```
#### Run the training script
**Make sure that $GARAGE is in your $PYTHONPATH** \
Ex.

    $ python $GARAGE/contrib/ros/envs/example_launchers/trpo_gazebo_sawyer_pnp.py
## Troubleshooting
### ImportError: import cv2 when running training script
Make sure your are using opencv-python installed in garage conda env.
