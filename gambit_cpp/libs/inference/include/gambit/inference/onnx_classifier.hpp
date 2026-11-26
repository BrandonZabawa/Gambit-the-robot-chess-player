// Square classifier wrapper: 64 square crops in, (64, 13) class scores out.
// 13 classes = empty + 6 white + 6 black pieces (order defined by gambit_vision's labels
// notebook; keep in sync). Built against ONNX Runtime when GAMBIT_WITH_ONNX=ON, otherwise a
// stub whose run() throws, so the rest of the stack still links.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gambit::inference {

inline constexpr std::size_t kNumSquares = 64;
inline constexpr std::size_t kNumClasses = 13;

struct CropBatch {
  // NCHW float32, N = 64, normalized exactly as in training (see gambit_vision).
  int channels = 3;
  int height = 0;
  int width = 0;
  std::vector<float> data;  // size 64 * C * H * W
};

using Scores = std::vector<std::array<float, kNumClasses>>;  // 64 rows

class OnnxClassifier {
 public:
  OnnxClassifier();
  ~OnnxClassifier();
  OnnxClassifier(OnnxClassifier&&) noexcept;
  OnnxClassifier& operator=(OnnxClassifier&&) noexcept;

  static bool available();  // true when compiled with ONNX Runtime

  // Load an exported model (e.g. models/square_classifier.onnx). Throws on failure.
  void load(const std::string& model_path);
  bool loaded() const;

  // Run all 64 crops in one batch; returns raw logits (64, 13).
  Scores run(const CropBatch& crops);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gambit::inference
