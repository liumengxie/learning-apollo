cmake_minimum_required(VERSION 3.8)
project(control_pkg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)

set(msg_files
  msg/TrajectoryPoint.msg
  msg/Trajectory.msg
  msg/Localization.msg
  msg/Chassis.msg
  msg/ControlCommand.msg
  msg/DebugInfo.msg
)
rosidl_generate_interfaces(${PROJECT_NAME} ${msg_files} DEPENDENCIES std_msgs)

add_library(control_core SHARED
  src/vehicle_model.cpp src/anomaly_detector.cpp src/mpc_controller.cpp)
target_include_directories(control_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(control_core PUBLIC Eigen3::Eigen)

add_executable(control_node src/control_node.cpp)
target_include_directories(control_node PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(control_node control_core ${rosidl_TARGETS})
ament_target_dependencies(control_node rclcpp std_msgs)
rosidl_target_interfaces(control_node ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS control_core ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS control_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  add_subdirectory(test)
endif()

ament_package()