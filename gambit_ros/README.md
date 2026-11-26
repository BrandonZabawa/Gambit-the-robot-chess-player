# gambit_ros

ROS 2 (C++, `rclcpp`) nodes that connect the camera, the game logic and the arm. Targets ROS 2
Lyrical (tested APIs are the same on Jazzy). Setup code: the plumbing (topics, parameters,
action client, debouncing) is real, while the chess/vision/IK logic is marked `TODO`.

## Build

```bash
# in a colcon workspace whose src/ contains (or symlinks) this repo
source /opt/ros/lyrical/setup.bash
rosdep install --from-paths src --ignore-src -y
colcon build --packages-select gambit_description gambit_bringup gambit_ros
source install/setup.bash
ros2 launch gambit_bringup gambit.launch.py start_nodes:=true
```

You can also run each node on its own:
`ros2 run gambit_ros vision_node --ros-args --params-file install/gambit_ros/share/gambit_ros/config/params.yaml`.

## How the nodes connect

```
Gazebo / real camera --/camera/image--> vision_node --/gambit/board_fen--> game_node
game_node --/gambit/robot_move--> arm_node --FollowJointTrajectory--> /arm_controller (ros2_control)
arm_node --/gambit/arm_status--> (UI, game_node later)
```

| Topic / action | Type | Publisher -> subscriber |
|---|---|---|
| `/camera/image` | sensor_msgs/msg/Image | ros_gz_bridge (sim) or camera driver -> vision_node |
| `/camera/camera_info` | sensor_msgs/msg/CameraInfo | ros_gz_bridge -> (calibration checks) |
| `/gambit/board_fen` | std_msgs/msg/String (placement FEN, transient local) | vision_node -> game_node |
| `/gambit/robot_move` | std_msgs/msg/String (UCI, e.g. `e7e5`) | game_node -> arm_node |
| `/gambit/arm_status` | std_msgs/msg/String (`moving`/`done`/`error`) | arm_node -> anyone |
| `/arm_controller/follow_joint_trajectory` | control_msgs/action/FollowJointTrajectory | arm_node -> joint_trajectory_controller |
| `/gripper_controller/follow_joint_trajectory` | control_msgs/action/FollowJointTrajectory | arm_node (TODO) -> gripper controller |
| `/world/gambit_chess/set_pose` | ros_gz_interfaces/srv/SetEntityPose | sim tools (gambit_sim teleport) -> Gazebo |

Names are in `include/gambit_ros/topics.hpp`; parameters in `config/params.yaml`.

## Nodes

* **vision_node** (`src/vision_node.cpp`): classifies every n-th frame and publishes the FEN only
  after `stable_frames` identical, confident predictions. TODO: ArUco -> homography -> crops ->
  classifier. The classifier is meant to come from the **gambit_cpp** library (another package in
  this repo). Assumed interface: CMake package `gambit_cpp` exporting target `gambit_cpp::gambit_cpp`,
  header `gambit_cpp/onnx_classifier.hpp`, class `gambit::OnnxClassifier(const std::string& onnx_path)`
  with `predict_proba(crops) -> 64 x 13 probabilities` (contracts.md order). If CMake can't find
  it, the node builds as a stub (`GAMBIT_HAVE_CPP` undefined).
* **game_node** (`src/game_node.cpp`): keeps the game, infers the human move from two placements
  (move-detection gate), asks the engine (Stockfish over UCI) and publishes the reply. TODO: rules and engine.
* **arm_node** (`src/arm_node.cpp`): turns a UCI move into trajectory goals; the action client
  and result handling are real (it currently sends the park pose). TODO: pick-and-place waypoints with IK, gripper goals.

## Later: MoveIt 2

Hand-written IK + single trajectories are enough for a fixed board. When you need collision
avoidance (arm vs. pieces/camera) or Cartesian paths, generate a MoveIt 2 config for the SO-ARM101
(`moveit_setup_assistant` on `gambit_description/urdf/so101.urdf.xacro`) and replace
`send_trajectory` with `moveit::planning_interface::MoveGroupInterface` (or MoveIt Servo for
gamepad jogging with `gambit_bringup/config/joy_mapping.yaml`).
