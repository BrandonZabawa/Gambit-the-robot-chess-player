# Gambit C++ layer: servo bus, kinematics, square map, ONNX inference, arm daemon.
# Build:  cmake -S gambit_cpp -B build -G Ninja && cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.24)
project(gambit_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(GAMBIT_BUILD_TESTS  "Build GoogleTest unit tests"                      ON)
option(GAMBIT_BUILD_PYTHON "Build the pybind11 module (gambit_cpp_py)"        OFF)
option(GAMBIT_WITH_ONNX    "Link ONNX Runtime (set ONNXRUNTIME_ROOT)"         OFF)
option(GAMBIT_BUILD_DAEMON "Build the gambit_arm_daemon executable"           ON)

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# ---------------------------------------------------------------- Eigen
# Prefer the system package (apt install libeigen3-dev); fall back to a shallow git clone.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen3 not found on the system: fetching it with FetchContent")
  # Header-only: SOURCE_SUBDIR points at a folder with no CMakeLists.txt, so MakeAvailable only
  # downloads it and we wrap the headers in an INTERFACE target (skips Eigen's own CMake).
  FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  gambit_headers_only)
  FetchContent_MakeAvailable(eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
  target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
endif()

# ---------------------------------------------------------------- spdlog
find_package(spdlog QUIET)
if(NOT TARGET spdlog::spdlog)
  FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(spdlog)
endif()

# ---------------------------------------------------------------- ONNX Runtime (optional)
if(GAMBIT_WITH_ONNX)
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime install prefix")
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "GAMBIT_WITH_ONNX=ON but ONNX Runtime not found; set ONNXRUNTIME_ROOT")
  endif()
endif()

# ---------------------------------------------------------------- libraries
add_library(gambit_servo_bus
  libs/servo_bus/src/sts_protocol.cpp
  libs/servo_bus/src/serial_port.cpp)
target_include_directories(gambit_servo_bus PUBLIC libs/servo_bus/include)
target_compile_options(gambit_servo_bus PRIVATE -Wall -Wextra)

add_library(gambit_kinematics libs/kinematics/src/so101.cpp)
target_include_directories(gambit_kinematics PUBLIC libs/kinematics/include)
target_link_libraries(gambit_kinematics PUBLIC Eigen3::Eigen)
target_compile_options(gambit_kinematics PRIVATE -Wall -Wextra)

add_library(gambit_square_map libs/square_map/src/square_map.cpp)
target_include_directories(gambit_square_map PUBLIC libs/square_map/include)
target_compile_options(gambit_square_map PRIVATE -Wall -Wextra)

add_library(gambit_inference libs/inference/src/onnx_classifier.cpp)
target_include_directories(gambit_inference PUBLIC libs/inference/include)
target_compile_options(gambit_inference PRIVATE -Wall -Wextra)
if(GAMBIT_WITH_ONNX)
  target_compile_definitions(gambit_inference PUBLIC GAMBIT_WITH_ONNX=1)
  target_include_directories(gambit_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(gambit_inference PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

foreach(lib gambit_servo_bus gambit_kinematics gambit_square_map gambit_inference)
  set_target_properties(${lib} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

# ---------------------------------------------------------------- daemon
if(GAMBIT_BUILD_DAEMON)
  add_executable(gambit_arm_daemon app/gambit_arm_daemon.cpp)
  target_link_libraries(gambit_arm_daemon PRIVATE
    gambit_servo_bus gambit_kinematics gambit_square_map spdlog::spdlog)
endif()

# ---------------------------------------------------------------- python bindings (optional)
if(GAMBIT_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(gambit_cpp_py bindings/pybind_module.cpp)
  target_link_libraries(gambit_cpp_py PRIVATE
    gambit_servo_bus gambit_kinematics gambit_square_map gambit_inference)
endif()

# ---------------------------------------------------------------- tests
if(GAMBIT_BUILD_TESTS)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  add_executable(gambit_tests
    tests/test_sts_protocol.cpp
    tests/test_square_map.cpp
    tests/test_kinematics.cpp
    tests/test_inference.cpp)
  target_link_libraries(gambit_tests PRIVATE
    gambit_servo_bus gambit_kinematics gambit_square_map gambit_inference GTest::gtest_main)
  gtest_discover_tests(gambit_tests)
endif()
