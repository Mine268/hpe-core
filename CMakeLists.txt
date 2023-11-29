cmake_minimum_required(VERSION 3.5)

# TODO
set(OpenCV_DIR "E:/opencv-4.8.0/")

project(hpe_core)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("include/")

# 人体姿态估计核心
add_library(hpe_core "src/hpe_core.cpp")
target_link_libraries(hpe_core PRIVATE ${OpenCV_LIBS})

# 使用 COCO wholebody 数据集的 hrnet 网络的二维人体姿态估计机器
add_library(hrnet_2d "src/hrnet_2d.cpp")
target_link_directories(hrnet_2d PRIVATE hpe_core ${OpenCV_LIBS})

# 便于调试用的 main
add_executable(hpe_main "src/main.cpp")
target_link_libraries(hpe_main PRIVATE hrnet_2d ${OpenCV_LIBS})
