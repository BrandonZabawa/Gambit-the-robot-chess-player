#include "gambit/inference/onnx_classifier.hpp"

#include <stdexcept>

#ifdef GAMBIT_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace gambit::inference {

#ifdef GAMBIT_WITH_ONNX
struct OnnxClassifier::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "gambit"};
  std::unique_ptr<Ort::Session> session;
};
bool OnnxClassifier::available() { return true; }
#else
struct OnnxClassifier::Impl {
  bool loaded = false;
};
bool OnnxClassifier::available() { return false; }
#endif

OnnxClassifier::OnnxClassifier() : impl_(std::make_unique<Impl>()) {}
OnnxClassifier::~OnnxClassifier() = default;
OnnxClassifier::OnnxClassifier(OnnxClassifier&&) noexcept = default;
OnnxClassifier& OnnxClassifier::operator=(OnnxClassifier&&) noexcept = default;

void OnnxClassifier::load(const std::string& model_path) {
#ifdef GAMBIT_WITH_ONNX
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  // TODO: choose execution provider (CUDA/TensorRT on Jetson Orin), graph optimization level.
  impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), opts);
#else
  (void)model_path;
  throw std::runtime_error("built without ONNX Runtime (configure with -DGAMBIT_WITH_ONNX=ON)");
#endif
}

bool OnnxClassifier::loaded() const {
#ifdef GAMBIT_WITH_ONNX
  return impl_->session != nullptr;
#else
  return impl_->loaded;
#endif
}

Scores OnnxClassifier::run(const CropBatch& crops) {
  const std::size_t expected =
      kNumSquares * static_cast<std::size_t>(crops.channels * crops.height * crops.width);
  if (crops.data.size() != expected || expected == 0)
    throw std::invalid_argument("CropBatch.data must hold 64 * C * H * W floats");
#ifdef GAMBIT_WITH_ONNX
  if (!impl_->session) throw std::logic_error("model not loaded");
  // TODO: build the input tensor (shape {64, C, H, W}), look up input/output names from the
  // session, call session->Run, check the output shape is {64, 13} and copy it into Scores.
  throw std::logic_error("OnnxClassifier::run: TODO");
#else
  throw std::runtime_error("built without ONNX Runtime (configure with -DGAMBIT_WITH_ONNX=ON)");
#endif
}

}  // namespace gambit::inference
