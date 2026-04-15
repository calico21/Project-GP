cmake_minimum_required(VERSION 3.8)
project(mirena_common)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)  # Add if your nodes use RCLCPP
find_package(rosidl_default_generators REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# ================== Message/Service Generation ==================
set(COMMON_MSG
  "msg/ASStatus.msg"
  "msg/MissionStatus.msg"
  "msg/BezierCurve.msg" 
  "msg/BezierCurvePoint.msg" 
  "msg/CarControl.msg" 
  "msg/CarParameters.msg" 
  "msg/Entity.msg"
  "msg/DebugEntity.msg"
  "msg/EntityList.msg"
  "msg/DebugEntityList.msg"
  "msg/Car.msg"
  "msg/WheelSpeeds.msg"
  "msg/FlatPoseDist.msg"
  "msg/Gate.msg" 
  "msg/Track.msg"
  "msg/Empty.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${COMMON_MSG}
  ${COMMON_SRV}
  DEPENDENCIES geometry_msgs
)


# ================== Header Includer ==================

# Install const header files explicitly
install(
  DIRECTORY const/
  DESTINATION include/${PROJECT_NAME}/${PROJECT_NAME}/const
  FILES_MATCHING PATTERN "mirena_const.h"
)

# ================== Common Library Configuration ==================

set(mirena_lib_sources
  "src/common_lib/bezier_curve/bezier_curve.cpp"
  "src/conversion/ros_conversion.cpp"
  "src/debug/debug.cpp"
)

# Add library target
add_library(mirena_lib STATIC ${mirena_lib_sources})

# Include directories - both for build and install
target_include_directories(mirena_lib
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}/${PROJECT_NAME}>
)

# Dependencies
target_link_libraries(mirena_lib PRIVATE 
  $<BUILD_INTERFACE:tf2_geometry_msgs::tf2_geometry_msgs>
)

set(MIRENA_LIB_DEPENDENCIES
  geometry_msgs
  rclcpp
  visualization_msgs
)

ament_target_dependencies(mirena_lib
  PUBLIC
  ${MIRENA_LIB_DEPENDENCIES}
)

# Message typesupport
rosidl_get_typesupport_target(
  typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)
target_link_libraries(mirena_lib PRIVATE ${typesupport_target})

# ================== Installation ==================
install(TARGETS mirena_lib 
  EXPORT export_mirena_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers with proper structure
install(
  DIRECTORY src/
  DESTINATION include/${PROJECT_NAME}/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Export targets
install(EXPORT export_mirena_lib 
  FILE "mirenaLibTargets.cmake"
  DESTINATION share/${PROJECT_NAME}/cmake
  NAMESPACE ${PROJECT_NAME}::
)

# Ament specific exports
ament_export_targets(export_mirena_lib)
ament_export_include_directories(include/${PROJECT_NAME}/${PROJECT_NAME})
ament_export_dependencies(${MIRENA_LIB_DEPENDENCIES})

# ================== Finalize ==================
ament_package()
