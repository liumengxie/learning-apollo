cmake_minimum_required(VERSION 3.8)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})

add_executable(test_vehicle_model test_vehicle_model.cpp)
target_link_libraries(test_vehicle_model control_core GTest::GTest GTest::Main pthread)
add_test(NAME test_vehicle_model COMMAND test_vehicle_model)

add_executable(test_anomaly_detector test_anomaly_detector.cpp)
target_link_libraries(test_anomaly_detector control_core GTest::GTest GTest::Main pthread)
add_test(NAME test_anomaly_detector COMMAND test_anomaly_detector)

add_executable(test_mpc_controller test_mpc_controller.cpp)
target_link_libraries(test_mpc_controller control_core GTest::GTest GTest::Main pthread)
add_test(NAME test_mpc_controller COMMAND test_mpc_controller)