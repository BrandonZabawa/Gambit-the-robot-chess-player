// arm_node: /gambit/robot_move (UCI) -> pick-and-place as FollowJointTrajectory goals.
//
// For a move "e7e5": (capture? remove the captured piece to the graveyard first) ->
// hover over e7 -> descend -> close gripper -> lift -> hover over e5 -> descend -> open -> lift
// -> park. Joint targets come from inverse kinematics of the SO-ARM101 (gambit_arm notebooks;
// MoveIt 2 later). Square positions come from gambit_bringup/config/board.yaml.
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

#include "gambit_ros/chess_utils.hpp"
#include "gambit_ros/topics.hpp"

namespace gambit_ros
{

using FollowJT = control_msgs::action::FollowJointTrajectory;
using GoalHandle = rclcpp_action::ClientGoalHandle<FollowJT>;

class ArmNode : public rclcpp::Node
{
public:
  ArmNode()
  : Node("arm_node")
  {
    const auto move_topic = declare_parameter<std::string>("robot_move_topic", topics::kRobotMove);
    const auto arm_action = declare_parameter<std::string>("arm_action", topics::kArmAction);
    joints_ = declare_parameter<std::vector<std::string>>(
      "joints", {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"});
    park_ = declare_parameter<std::vector<double>>("park", {-1.6, -1.4, 1.3, 1.2, 0.0});
    segment_time_s_ = declare_parameter<double>("segment_time_s", 2.0);

    arm_client_ = rclcpp_action::create_client<FollowJT>(this, arm_action);
    status_pub_ = create_publisher<std_msgs::msg::String>(topics::kArmStatus, 10);
    move_sub_ = create_subscription<std_msgs::msg::String>(
      move_topic, 10, [this](std_msgs::msg::String::ConstSharedPtr msg) {on_move(msg->data);});
    RCLCPP_INFO(get_logger(), "arm_node: waiting for moves on %s", move_topic.c_str());
  }

private:
  void on_move(const std::string & /*uci*/)
  {
    // Design:
    // - parse_uci the message; malformed -> log an error and ignore it.
    // - If a trajectory is still running (busy_), drop the move with a warning (or queue it?
    //   decide: dropping is safer, the game node can resend).
    // - Build the pick-and-place waypoint list from the file header: capture first (captured
    //   piece to the graveyard), then hover / descend / grip / lift / move / descend / release /
    //   lift / park. Square xyz from board.yaml, joint angles from the SO-ARM101 IK (gambit_arm).
    // - Gripper open/close are separate goals on topics::kGripperAction between arm segments.
    // - First milestone: send only the park pose, to test the action plumbing end to end.
    // TODO: implement.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: on_move");
  }

  void send_trajectory(const std::vector<std::vector<double>> & /*waypoints*/)
  {
    // Design:
    // - wait_for_action_server with a short timeout; unavailable -> status "error".
    // - FollowJointTrajectory goal: joint_names = joints_, one JointTrajectoryPoint per waypoint,
    //   time_from_start accumulating segment_time_s_ per segment (rclcpp::Duration::from_seconds).
    // - SendGoalOptions: goal_response_callback (null handle = rejected -> error, clear busy_),
    //   result_callback (SUCCEEDED and error_code == SUCCESSFUL -> "done", else "error"; clear busy_).
    // - Set busy_ and publish "moving" before async_send_goal.
    // - On error: stop and ask the human; never retry blindly next to a person's hand.
    // TODO: implement.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: send_trajectory");
  }

  void publish_status(const std::string & /*s*/)
  {
    // Design: wrap the string (idle | moving | done | error) in std_msgs/String and publish on status_pub_.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: publish_status");
  }

  std::vector<std::string> joints_;
  std::vector<double> park_;
  double segment_time_s_{2.0};
  bool busy_{false};
  rclcpp_action::Client<FollowJT>::SharedPtr arm_client_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr move_sub_;
};

}  // namespace gambit_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gambit_ros::ArmNode>());
  rclcpp::shutdown();
  return 0;
}
