// Test outlines for SO-101 kinematics (same numbers as kinematics.ipynb).
#include <gtest/gtest.h>

#include "gambit/kinematics/so101.hpp"

using namespace gambit::kinematics;

TEST(Kinematics, ZeroPoseIsFinite) {
  // forward(Zero) is all finite and its rotation part is orthonormal
  // (|R R^T - I| < 1e-5).
  GTEST_SKIP() << "TODO";
}

TEST(Kinematics, ZeroPoseMatchesNotebook) {
  // kinematics.ipynb: fk(zeros) tool position ~ (0.3914, 0.0, 0.2265) m, tolerance 0.5 mm.
  GTEST_SKIP() << "TODO";
}

TEST(Kinematics, PanRotatesAboutVertical) {
  // The pan axis is vertical through the shoulder_pan origin (x = 0.0388 m, y = 0). With
  // q[0] = 0.5: tool height unchanged and horizontal distance from that axis unchanged
  // (1e-4), and y < 0 (origin rpy (pi, 0, -pi) flips the axis: +pan swings toward -y).
  GTEST_SKIP() << "TODO";
}

TEST(Kinematics, JacobianShapeAndFinite) {
  // numeric_jacobian at q = (0.1, -0.3, 0.4, 1.2, 0): all finite, position rows nonzero.
  // Optional: compare against the notebook's numeric_jacobian at the same q.
  GTEST_SKIP() << "TODO";
}

TEST(Kinematics, InverseRoundTrip) {
  // Once inverse() is ported: FK of a known q -> inverse(position, approach, q + 0.1)
  // converges and the FK of the result is within 1 mm of the original tip.
  // Until then, assert it returns nullopt for an unreachable target.
  GTEST_SKIP() << "TODO";
}
