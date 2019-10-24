cmake_minimum_required(VERSION 2.8.3)
project(arlorobot_cartographer)

find_package(catkin REQUIRED)

catkin_package(CATKIN_DEPENDS
    cartographer
    cartographer_ros
)
