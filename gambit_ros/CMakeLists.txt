cmake_minimum_required(VERSION 3.8)
project(gambit_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Optional: the gambit_cpp library (OnnxClassifier for the square classifier) and OpenCV
# (warping / cropping). Without them vision_node builds as a stub that publishes nothing useful.
find_package(gambit_cpp QUIET)
find_package(OpenCV QUIET COMPONENTS core imgproc)

add_executable(vision_node src/vision_node.cpp)
target_include_directories(vision_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(vision_node PUBLIC
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  ${sensor_msgs_TARGETS})
if(gambit_cpp_FOUND)
  target_link_libraries(vision_node PUBLIC gambit_cpp::gambit_cpp)
  target_compile_definitions(vision_node PRIVATE GAMBIT_HAVE_CPP=1)
endif()
if(OpenCV_FOUND)
  target_link_libraries(vision_node PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(vision_node PRIVATE GAMBIT_HAVE_OPENCV=1)
endif()

add_executable(game_node src/game_node.cpp)
target_include_directories(game_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(game_node PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS})

add_executable(arm_node src/arm_node.cpp)
target_include_directories(arm_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(arm_node PUBLIC
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  ${std_msgs_TARGETS}
  ${control_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${builtin_interfaces_TARGETS})

install(TARGETS vision_node game_node arm_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
