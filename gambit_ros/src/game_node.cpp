// game_node: /gambit/board_fen -> decide -> /gambit/robot_move (UCI).
//
// Keeps the game state, infers the human's move from two consecutive board FENs (the move-
// detection gate: exactly one legal move must turn the old placement into the new one), asks
// the engine for the robot's reply and publishes it. The chess rules and the engine live in
// gambit_game (Python notebooks today); here they are TODOs, e.g. port with a small C++ chess
// library or call Stockfish over UCI directly (boost::process / popen).
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "gambit_ros/chess_utils.hpp"
#include "gambit_ros/topics.hpp"

namespace gambit_ros
{

class GameNode : public rclcpp::Node
{
public:
  GameNode()
  : Node("game_node")
  {
    const auto fen_topic = declare_parameter<std::string>("board_fen_topic", topics::kBoardFen);
    const auto move_topic = declare_parameter<std::string>("robot_move_topic", topics::kRobotMove);
    robot_color_ = declare_parameter<std::string>("robot_color", "black");
    engine_path_ = declare_parameter<std::string>("engine_path", "stockfish");
    think_time_s_ = declare_parameter<double>("think_time_s", 0.5);
    elo_ = declare_parameter<int>("elo", 1700);

    move_pub_ = create_publisher<std_msgs::msg::String>(move_topic, 10);
    fen_sub_ = create_subscription<std_msgs::msg::String>(
      fen_topic, rclcpp::QoS(10).transient_local(),
      [this](std_msgs::msg::String::ConstSharedPtr msg) {on_board(msg->data);});
    current_placement_ = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
    RCLCPP_INFO(get_logger(), "game_node: robot plays %s", robot_color_.c_str());
  }

private:
  void on_board(const std::string & placement)
  {
    if (!placement_from_fen(placement)) {
      RCLCPP_WARN(get_logger(), "ignoring malformed FEN '%s'", placement.c_str());
      return;
    }
    if (placement == current_placement_) {return;}
    if (placement == expected_after_robot_) {
      // The robot's own move has been executed and seen by the camera.
      current_placement_ = placement;
      expected_after_robot_.clear();
      return;
    }
    const std::string human_move = infer_move(current_placement_, placement);
    if (human_move.empty()) {
      RCLCPP_WARN(get_logger(), "board changed but no single legal move explains it; waiting");
      return;   // TODO: publish a "please fix the board" status for the UI
    }
    current_placement_ = placement;
    const std::string reply = choose_move();
    if (reply.empty()) {return;}
    std_msgs::msg::String out;
    out.data = reply;
    move_pub_->publish(out);
    RCLCPP_INFO(get_logger(), "human %s -> robot %s", human_move.c_str(), reply.c_str());
  }

  // TODO: return the UCI move that turns `before` into `after` (legal in the full game state:
  // side to move, castling rights, en passant), or "" if none / ambiguous. Keep a full move
  // history (not just placements) so castling and en passant work.
  std::string infer_move(const std::string & /*before*/, const std::string & /*after*/)
  {
    return "";
  }

  // TODO: run the engine (UCI: "position startpos moves ...", "go movetime <ms>", read "bestmove"),
  // with UCI_LimitStrength/UCI_Elo = elo_; set expected_after_robot_ to the placement after the move.
  std::string choose_move()
  {
    return "";
  }

  std::string robot_color_, engine_path_;
  double think_time_s_{0.5};
  int elo_{1700};
  std::string current_placement_, expected_after_robot_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr move_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fen_sub_;
};

}  // namespace gambit_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gambit_ros::GameNode>());
  rclcpp::shutdown();
  return 0;
}
