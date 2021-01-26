cmake_minimum_required(VERSION 3.5)
project(stereoVision)

find_package(Pangolin REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11")
include_directories("/usr/include/eigen3")

find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})