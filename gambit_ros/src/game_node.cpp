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
  void on_board(const std::string & /*placement*/)
  {
    // Design:
    // - Validate with placement_from_fen; malformed -> warn and ignore.
    // - Unchanged placement -> nothing to do. Equal to expected_after_robot_ -> the robot's own
    //   move has been executed and seen: accept it as current and clear the expectation.
    // - Otherwise infer the human's move (move-detection gate: exactly one legal move explains the
    //   change). None / ambiguous -> warn, wait, and (later) publish a "please fix the board" status.
    // - Accepted move -> update the game state, choose_move(), publish the reply UCI on move_pub_.
    // Why the gate: the vision publishes intermediate boards (hand over the board, half-done
    // move); only a board that is one legal move away is trusted.
    // TODO: implement.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: on_board");
  }

  // Return the UCI move that turns `before` into `after`, or "" if none / ambiguous.
  std::string infer_move(const std::string & /*before*/, const std::string & /*after*/)
  {
    // Design:
    // - Needs the full game state (side to move, castling rights, en passant), so keep a move
    //   history, not just placements.
    // - Candidates = all legal moves; apply each and compare the resulting placement with `after`.
    //   Exactly one match -> that move. Same idea as gambit_game's infer_move (Python).
    // Decisions for you: port a small C++ chess library, or ask the Python side / engine.
    // TODO: implement.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: infer_move");
    return "";
  }

  // Engine reply as UCI, or "" on failure.
  std::string choose_move()
  {
    // Design:
    // - Talk UCI to engine_path_ (popen / boost::process): "uci", setoption UCI_LimitStrength
    //   true + UCI_Elo elo_, "position startpos moves ...", "go movetime <think_time_s_ in ms>",
    //   read until "bestmove".
    // - Set expected_after_robot_ to the placement after that move, so on_board recognizes it.
    // TODO: implement.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: choose_move");
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
