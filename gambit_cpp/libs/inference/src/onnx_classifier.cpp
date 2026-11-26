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

// Construction only allocates the pimpl (layout, not logic); destructor / moves are defaulted.
OnnxClassifier::OnnxClassifier() : impl_(std::make_unique<Impl>()) {}
OnnxClassifier::~OnnxClassifier() = default;
OnnxClassifier::OnnxClassifier(OnnxClassifier&&) noexcept = default;
OnnxClassifier& OnnxClassifier::operator=(OnnxClassifier&&) noexcept = default;

void OnnxClassifier::load(const std::string& /*model_path*/) {
  // Design:
  // - With ONNX Runtime: Ort::SessionOptions (SetIntraOpNumThreads, graph optimisation
  //   level; execution provider CUDA / TensorRT on the Jetson Orin), then
  //   impl_->session = std::make_unique<Ort::Session>(env, path, opts). Ort throws
  //   Ort::Exception on a bad file; let it propagate.
  // - Without: throw std::runtime_error telling the user to configure -DGAMBIT_WITH_ONNX=ON
  //   (tests check this behaviour).
  throw std::logic_error("TODO: OnnxClassifier::load");
}

bool OnnxClassifier::loaded() const {
  // Design: session != nullptr (ONNX build) / impl_->loaded flag (stub build).
  throw std::logic_error("TODO: OnnxClassifier::loaded");
}

Scores OnnxClassifier::run(const CropBatch& /*crops*/) {
  // Design:
  // - Validate first: data.size() must equal 64 * C * H * W and be nonzero, else
  //   std::invalid_argument (cheap check that catches most integration bugs).
  // - ONNX build: logic_error if not loaded; Ort::Value::CreateTensor over crops.data with
  //   shape {64, C, H, W} (no copy); input / output names from the session; session->Run;
  //   check the output shape is {64, 13}; copy rows into Scores.
  // - Stub build: std::runtime_error as in load().
  // Preprocessing (resize, normalisation) must match gambit_vision training exactly; it is
  // the caller's job, documented on CropBatch.
  throw std::logic_error("TODO: OnnxClassifier::run");
}

}  // namespace gambit::inference
