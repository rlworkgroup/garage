cmake_minimum_required(VERSION 2.8.3)
project(sawyer_collision_avoidance)
set(CMAKE_CXX_STANDARD 11)

find_package(fcl REQUIRED)

set(fcl_INCLUDE_DIRS /usr/local/include/fcl)
## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
             roscpp
             rospy
             intera_core_msgs
)

include_directories(
        ${fcl_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

add_executable(sawyer_collision_avoidance_node main.cc sawyer_gripper.cc sawyer_gripper.h)

target_link_libraries(sawyer_collision_avoidance_node
        fcl
        ${catkin_LIBRARIES}
        ${Boost_LIBRARIES}
)




