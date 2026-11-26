// Test outlines for the ONNX classifier wrapper.
#include <gtest/gtest.h>

#include "gambit/inference/onnx_classifier.hpp"

using namespace gambit::inference;

TEST(Inference, RejectsWrongBatchSize) {
  // CropBatch with H = W = 8 but only 10 floats: run() throws std::invalid_argument
  // (validation happens before any ONNX call, so this works in both builds).
  GTEST_SKIP() << "TODO";
}

TEST(Inference, StubWithoutOnnx) {
  // Only when !OnnxClassifier::available(): load("missing.onnx") throws
  // std::runtime_error and loaded() stays false.
  GTEST_SKIP() << "TODO";
}

TEST(Inference, TinyModelShape) {
  // With ONNX Runtime: export a tiny model from gambit_vision, run 64 random crops, assert
  // the output has 64 rows of 13 finite scores.
  GTEST_SKIP() << "TODO";
}
