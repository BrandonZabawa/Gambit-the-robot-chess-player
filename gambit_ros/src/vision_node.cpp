// vision_node: /camera/image -> board FEN on /gambit/board_fen.
//
// Pipeline (mirrors gambit_vision, see gambit_vision/docs/contracts.md):
//   image -> ArUco corners -> homography -> warped board -> 64 square crops (100 x 200)
//   -> square classifier (ONNX ResNet-18, 13 classes) -> (64, 13) probs -> board assembly
//   -> publish the placement FEN only when every square is confident and the result is stable.
//
// The classifier comes from the gambit_cpp library (another package in this repo):
// gambit::OnnxClassifier wraps onnxruntime. Built only when CMake finds gambit_cpp
// (GAMBIT_HAVE_CPP); otherwise this node runs as a stub that logs frames.
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

#include "gambit_ros/topics.hpp"

#ifdef GAMBIT_HAVE_CPP
#include "gambit_cpp/onnx_classifier.hpp"   // assumed interface, see README.md
#endif

namespace gambit_ros
{

class VisionNode : public rclcpp::Node
{
public:
  VisionNode()
  : Node("vision_node")
  {
    image_topic_ = declare_parameter<std::string>("image_topic", topics::kCameraImage);
    fen_topic_ = declare_parameter<std::string>("board_fen_topic", topics::kBoardFen);
    model_path_ = declare_parameter<std::string>("model_path", "models/square_classifier.onnx");
    min_confidence_ = declare_parameter<double>("min_confidence", 0.6);
    process_every_n_ = declare_parameter<int>("process_every_n", 5);
    stable_frames_ = declare_parameter<int>("stable_frames", 3);

    fen_pub_ = create_publisher<std_msgs::msg::String>(fen_topic_, rclcpp::QoS(10).transient_local());
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {on_image(msg);});

#ifdef GAMBIT_HAVE_CPP
    // TODO: classifier_ = std::make_unique<gambit::OnnxClassifier>(model_path_);
#else
    RCLCPP_WARN(get_logger(), "built without gambit_cpp: no classifier, FEN will not be published");
#endif
    RCLCPP_INFO(get_logger(), "vision_node: %s -> %s", image_topic_.c_str(), fen_topic_.c_str());
  }

private:
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & /*msg*/)
  {
    // Design:
    // - Process only every process_every_n_-th frame (CPU budget); accept rgb8 / bgr8 only
    //   (throttled warning otherwise).
    // - fen = predict_fen(); "" means not confident -> skip.
    // - Debounce: publish only after stable_frames_ identical predictions in a row (hands,
    //   shadows) and only when it differs from the last published FEN. The publisher is
    //   transient_local, so a late-starting game_node still gets the latest board.
    // TODO: implement.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: on_image");
  }

  // Returns the placement FEN, or "" when the board cannot be read confidently.
  std::string predict_fen(const sensor_msgs::msg::Image & /*msg*/)
  {
    // Design (mirrors gambit_vision, contracts.md):
    //  1. wrap msg data as cv::Mat (CV_8UC3, step = msg.step); bgr8 -> rgb8 if needed
    //  2. detect the 4 ArUco markers (DICT_4X4_50, ids a1=0 h1=1 h8=2 a8=3) -> homography -> warp
    //  3. cut 64 crops (camera.yaml crop parameters, 100 x 200) in square order a1..h8
    //  4. classifier_ (gambit_cpp OnnxClassifier) -> (64, 13) probs, classes empty,P,N,B,R,Q,K,p,n,b,r,q,k
    //  5. any square's max prob < min_confidence_ -> "" (never publish a board it is unsure about)
    //  6. build the placement FEN (rank 8 first), sanity check one king per side
    // TODO: implement.
    RCLCPP_WARN_ONCE(get_logger(), "TODO: predict_fen");
    return "";
  }

  std::string image_topic_, fen_topic_, model_path_;
  double min_confidence_{0.6};
  int process_every_n_{5}, stable_frames_{3};
  long frame_count_{0};
  int stable_count_{0};
  std::string last_fen_, published_fen_;
#ifdef GAMBIT_HAVE_CPP
  std::unique_ptr<gambit::OnnxClassifier> classifier_;
#endif
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fen_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};

}  // namespace gambit_ros

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gambit_ros::VisionNode>());
  rclcpp::shutdown();
  return 0;
}
