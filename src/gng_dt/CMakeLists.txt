cmake_minimum_required(VERSION 3.10)
project(gng_dt)

# C++ 標準
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ビルドタイプ
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# ===== パッケージ検索 =====

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters) 
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ===== インクルードディレクトリ =====

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# ===== GNG-DT ライブラリ =====

add_library(gng_dt_core
  src/math_utils.cpp
  src/gng.cpp
)
target_include_directories(gng_dt_core
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  PUBLIC ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(gng_dt_core
  Eigen3::Eigen
)

# ===== Saliency ライブラリ =====

add_library(saliency_lib
  src/saliency.cpp
)
target_include_directories(saliency_lib
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# ===== ROS2 ノード: GNG =====

add_executable(gng_node
  src/gng_node.cpp
)
target_include_directories(gng_node
  PRIVATE ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(gng_node
  gng_dt_core
  Eigen3::Eigen
)
ament_target_dependencies(gng_node
  rclcpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  tf2
  nav_msgs
  pcl_conversions
  visualization_msgs
)

# ===== ROS2 ノード: Saliency Debug =====

add_executable(saliency_debug_node
  test/debug.cpp
)
target_include_directories(saliency_debug_node
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PCL_INCLUDE_DIRS}
)
target_link_libraries(saliency_debug_node
  saliency_lib
  ${PCL_LIBRARIES}
)
ament_target_dependencies(saliency_debug_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GNGNode.msg"
  "msg/GNGGraph.msg"
  "srv/GetGraph.srv"
  "srv/GetNearestNode.srv"
  DEPENDENCIES std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(gng_node ${cpp_typesupport_target})

# ===== ビルドオプション =====

if(NOT MSVC)
  target_compile_options(gng_dt_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(saliency_lib PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(gng_node PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(saliency_debug_node PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ===== インストール設定 =====

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS gng_dt_core saliency_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(TARGETS gng_node saliency_debug_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  OPTIONAL
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ===== ament パッケージ =====

ament_package()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")