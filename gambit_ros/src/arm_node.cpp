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
  void on_move(const std::string & uci)
  {
    const auto mv = parse_uci(uci);
    if (!mv) {
      RCLCPP_ERROR(get_logger(), "bad move '%s'", uci.c_str());
      return;
    }
    if (busy_) {
      RCLCPP_WARN(get_logger(), "busy, dropping move %s", uci.c_str());
      return;
    }
    // TODO: build the full pick-and-place waypoint list for mv (see file header) with IK, and
    // gripper goals on /gripper_controller/follow_joint_trajectory between segments.
    // For now: go to park, which exercises the action plumbing end to end.
    std::vector<std::vector<double>> waypoints = {park_};
    send_trajectory(waypoints);
  }

  void send_trajectory(const std::vector<std::vector<double>> & waypoints)
  {
    if (!arm_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_logger(), "arm action server not available");
      publish_status("error");
      return;
    }
    FollowJT::Goal goal;
    goal.trajectory.joint_names = joints_;
    double t = 0.0;
    for (const auto & q : waypoints) {
      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.positions = q;
      t += segment_time_s_;
      pt.time_from_start = rclcpp::Duration::from_seconds(t);
      goal.trajectory.points.push_back(pt);
    }

    rclcpp_action::Client<FollowJT>::SendGoalOptions opts;
    opts.goal_response_callback = [this](const GoalHandle::SharedPtr & handle) {
        if (!handle) {
          RCLCPP_ERROR(get_logger(), "trajectory goal rejected");
          busy_ = false;
          publish_status("error");
        }
      };
    opts.result_callback = [this](const GoalHandle::WrappedResult & result) {
        busy_ = false;
        const bool ok = result.code == rclcpp_action::ResultCode::SUCCEEDED &&
          result.result->error_code == FollowJT::Result::SUCCESSFUL;
        publish_status(ok ? "done" : "error");
        // TODO: on error, stop and ask the human (never retry blindly next to a person's hand).
      };
    busy_ = true;
    publish_status("moving");
    arm_client_->async_send_goal(goal, opts);
  }

  void publish_status(const std::string & s)
  {
    std_msgs::msg::String m;
    m.data = s;
    status_pub_->publish(m);
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
