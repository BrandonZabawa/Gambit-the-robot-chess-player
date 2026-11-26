#include <gtest/gtest.h>

#include "gambit/inference/onnx_classifier.hpp"

using namespace gambit::inference;

TEST(Inference, RejectsWrongBatchSize) {
  OnnxClassifier clf;
  CropBatch b;
  b.height = b.width = 8;
  b.data.resize(10);
  EXPECT_THROW(clf.run(b), std::invalid_argument);
}

TEST(Inference, StubWithoutOnnx) {
  if (OnnxClassifier::available()) GTEST_SKIP() << "built with ONNX Runtime";
  OnnxClassifier clf;
  EXPECT_THROW(clf.load("missing.onnx"), std::runtime_error);
  EXPECT_FALSE(clf.loaded());
}
