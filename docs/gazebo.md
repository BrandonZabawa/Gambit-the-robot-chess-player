# Running Gambit in Gazebo

Tested headless on ROS 2 Jazzy with Gazebo Harmonic 8.10 (Nov 18, 2026): the world loads,
the camera publishes, the arm spawns, all three controllers activate, a joint trajectory
reaches its goal, and `set_pose` teleports a piece. Your setup is ROS 2 Lyrical with Gazebo
Jetty; the file formats are the same, but run these steps once to confirm.

## Files

| File | What it is |
|---|---|
| `gambit_bringup/world/chess_world.sdf` | World `gambit_chess`: table, board, camera, 32 pieces plus 2 spare queens per color |
| `gambit_bringup/models/chess_board/` | Board with ArUco corner markers (DICT_4X4_50, ids 0 to 3 at a1, h1, h8, a8) |
| `gambit_bringup/models/pieces/` | 12 piece models (`gambit_piece_wP` ... `gambit_piece_bK`), static by default |
| `gambit_bringup/models/board_camera/` | Fixed 640 x 480 camera; topics `/camera/image`, `/camera/camera_info` |
| `gambit_bringup/models/aruco_markers/` | The same markers as PNGs to print for the real board |
| `gambit_bringup/config/board.yaml` | The measurements the world was built from (square 25 mm, board center 215 mm from the arm) |
| `gambit_bringup/config/controllers.yaml` | `joint_state_broadcaster`, `arm_controller`, `gripper_controller` (joint trajectory, position) |
| `gambit_bringup/config/gz_bridge.yaml` | Bridge: clock, camera image, camera info |
| `gambit_description/urdf/so101.urdf.xacro` | SO-ARM101 bolted to the world at the origin, with ros2_control and the gz_ros2_control plugin |
| `gambit_description/meshes/` | Arm meshes (TheRobotStudio SO-ARM100, Apache 2.0) |

Piece model names in the world: `w_P_1` to `w_P_8`, `w_N_1`, `w_N_2`, ..., `b_K_1`; spare
queens `w_Q_2`, `w_Q_3`, `b_Q_2`, `b_Q_3` start in the graveyard at (0.17, -0.175).

Board layout: world origin at the arm base, x toward the board, y to the arm's left. Square
centers: x = 0.215 + (rank - 3.5) * 0.025, y = (3.5 - file) * 0.025, with file 0 = a and
rank 0 = rank 1, on the board top at z = 0.006.

## Install (Lyrical)

```bash
sudo apt install ros-lyrical-ros-gz ros-lyrical-gz-ros2-control ros-lyrical-ros2-controllers \
                 ros-lyrical-robot-state-publisher ros-lyrical-xacro
```

## Run by hand (5 terminals, each with `source /opt/ros/lyrical/setup.bash`)

From the repo root:

```bash
# 1. Gazebo (add -s --headless-rendering for no window)
export GZ_SIM_RESOURCE_PATH=$PWD/gambit_bringup/models:$PWD/gambit_bringup/models/pieces:$PWD
gz sim -r gambit_bringup/world/chess_world.sdf

# 2. Robot description
xacro gambit_description/urdf/so101.urdf.xacro use_sim:=true \
      controllers_file:=$PWD/gambit_bringup/config/controllers.yaml > /tmp/gambit.urdf
python3 -c "import yaml; yaml.safe_dump({'robot_state_publisher': {'ros__parameters': {'robot_description': open('/tmp/gambit.urdf').read(), 'use_sim_time': True}}}, open('/tmp/rsp.yaml', 'w'))"
ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file /tmp/rsp.yaml

# 3. Spawn the arm, then its controllers
ros2 run ros_gz_sim create -world gambit_chess -topic robot_description -name gambit_arm
ros2 run controller_manager spawner joint_state_broadcaster arm_controller gripper_controller

# 4. Bridge the camera and clock, and the teleport service
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$PWD/gambit_bringup/config/gz_bridge.yaml
ros2 run ros_gz_bridge parameter_bridge /world/gambit_chess/set_pose@ros_gz_interfaces/srv/SetEntityPose

# 5. Try it
ros2 action send_goal /arm_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll], points: [{positions: [0.0, 0.2, 0.3, 1.4, 0.0], time_from_start: {sec: 3}}]}}"
ros2 service call /world/gambit_chess/set_pose ros_gz_interfaces/srv/SetEntityPose \
  "{entity: {name: 'w_P_5', type: 2}, pose: {position: {x: 0.2025, y: -0.0125, z: 0.006}, orientation: {w: 1.0}}}"   # e2 pawn to e4
```

## Gotchas found while testing

* Passing the URDF to robot_state_publisher with `-p robot_description:="$(cat ...)"` crashes
  (the comments contain characters the argument parser rejects). Use a params file, as above,
  or your launch file.
* `GZ_SIM_RESOURCE_PATH` must include the repo root so `package://gambit_description/...`
  meshes resolve without building.
* The xacro's default `controllers_file` uses `$(find gambit_bringup)`, which needs
  gambit_bringup built as a package; until then pass `controllers_file:=` as above.
* Pieces are static (they move only by teleport). Set `<static>false</static>` in the piece
  models when you want physical grasping.
* The arm starts in its park pose: joints at the `initial_value` numbers in the xacro.
