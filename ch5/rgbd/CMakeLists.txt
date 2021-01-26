cmake_minimum_required(VERSION 3.0)
project(joinMap)

find_package(Pangolin REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})



# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
# Eigen
include_directories("/usr/include/eigen3")
target_link_libraries(joinMap Sophus::Sophus)




