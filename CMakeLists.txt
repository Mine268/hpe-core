cmake_minimum_required(VERSION 3.5)

project(hpe_core)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("include/")

# 人体姿态估计核心
add_library(hpe_core
        "src/hpe_core.cpp" # 接口
        "src/hrnet_2d.cpp" # hrnet二维姿态估计
        "src/triangulate_3d.cpp" # 三角化的三维人体姿态估计
)
target_link_libraries(hpe_core PRIVATE ${OpenCV_LIBS})

# 同步捕捉
add_library(sync_cap_core
        "src/sync_cap_core.cpp" # 同步捕捉
        "src/opencv_cap.cpp" # opencv的摄像头捕捉
)
target_link_libraries(sync_cap_core PRIVATE ${OpenCV_LIBS}})

# 便于调试用的 main
add_executable(hpe_main "src/main.cpp")
target_link_libraries(hpe_main PRIVATE hpe_core ${OpenCV_LIBS})
