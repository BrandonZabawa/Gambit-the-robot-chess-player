// Topic, action and service names shared by the gambit_ros nodes.
// Keep in sync with gambit_bringup/config/gz_bridge.yaml, controllers.yaml and README.md.
#pragma once

namespace gambit_ros
{
namespace topics
{
inline constexpr char kCameraImage[] = "/camera/image";            // sensor_msgs/msg/Image (Gazebo bridge or real camera)
inline constexpr char kCameraInfo[] = "/camera/camera_info";       // sensor_msgs/msg/CameraInfo
inline constexpr char kBoardFen[] = "/gambit/board_fen";           // std_msgs/msg/String, piece placement FEN
inline constexpr char kRobotMove[] = "/gambit/robot_move";         // std_msgs/msg/String, UCI move e.g. "e7e5"
inline constexpr char kArmStatus[] = "/gambit/arm_status";         // std_msgs/msg/String: idle | moving | done | error
inline constexpr char kArmAction[] = "/arm_controller/follow_joint_trajectory";
inline constexpr char kGripperAction[] = "/gripper_controller/follow_joint_trajectory";
inline constexpr char kSetPose[] = "/world/gambit_chess/set_pose";  // ros_gz_interfaces/srv/SetEntityPose (sim only)
}  // namespace topics
}  // namespace gambit_ros
