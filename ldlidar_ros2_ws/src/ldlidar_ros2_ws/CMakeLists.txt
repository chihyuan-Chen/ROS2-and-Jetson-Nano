cmake_minimum_required(VERSION 3.5)
project(ldlidar_sl_ros2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

enable_language(CUDA)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(CUDA 10.2 REQUIRED)
find_package(CUDA REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# user add
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/include/core/
  ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/include/dataprocess/
  ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/include/filter/
  ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/include/logger/
  ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/include/serialcom/
  /usr/local/cuda-10.2/include
)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
link_directories(/usr/local/cuda-10.2/include/)

file(GLOB  LDLIDAR_DRI_CORE ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/src/core/*.cpp)
file(GLOB  LDLIDAR_DRI_DATARPC ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/src/dataprocess/*.cpp)
file(GLOB  LDLIDAR_DRI_FILTER ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/src/filter/*.cpp)
file(GLOB  LDLIDAR_DRI_LOGGER ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/src/logger/*.cpp)
file(GLOB  LDLIDAR_DRI_SERIAL ${CMAKE_CURRENT_SOURCE_DIR}/ldlidar_driver/src/serialcom/*.cpp)

add_executable(${PROJECT_NAME}_node  
  ${CMAKE_CURRENT_SOURCE_DIR}/src/demo.cpp 
  ${LDLIDAR_DRI_CORE} 
  ${LDLIDAR_DRI_DATARPC} 
  ${LDLIDAR_DRI_FILTER} 
  ${LDLIDAR_DRI_LOGGER} 
  ${LDLIDAR_DRI_SERIAL}
)
ament_target_dependencies(${PROJECT_NAME}_node rclcpp sensor_msgs geometry_msgs)
#add_executable(listener src/subscriber_member_function.cpp)
cuda_add_executable(listener src/subscriber_member_function.cpp src/test/test.cu)

ament_target_dependencies(listener rclcpp sensor_msgs geometry_msgs std_msgs)


# Install 
install(TARGETS ${PROJECT_NAME}_node
 DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz2 scripts
  DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS
  listener
  DESTINATION lib/${PROJECT_NAME})

ament_package()

