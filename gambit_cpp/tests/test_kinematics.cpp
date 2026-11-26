#include <gtest/gtest.h>

#include "gambit/kinematics/so101.hpp"

using namespace gambit::kinematics;

TEST(Kinematics, ZeroPoseIsFinite) {
  const auto t = forward(JointVector::Zero());
  EXPECT_TRUE(t.matrix().allFinite());
  // Rotation part stays orthonormal.
  EXPECT_NEAR((t.linear() * t.linear().transpose() - Eigen::Matrix3d::Identity()).norm(), 0, 1e-5);
}

TEST(Kinematics, ZeroPoseMatchesNotebook) {
  // kinematics.ipynb: fk(zeros) tool position ~ (0.3914, 0.0, 0.2265) m.
  const auto p = forward(JointVector::Zero()).translation();
  EXPECT_NEAR(p.x(), 0.3914, 5e-4);
  EXPECT_NEAR(p.y(), 0.0, 5e-4);
  EXPECT_NEAR(p.z(), 0.2265, 5e-4);
}

TEST(Kinematics, PanRotatesAboutVertical) {
  JointVector q = JointVector::Zero();
  // The pan axis is vertical through the shoulder_pan origin (x = 0.0388 m in the base frame),
  // so panning keeps the tool's height and its distance from that axis.
  const Eigen::Vector2d axis(0.0388353, 0.0);
  const auto p0 = forward(q).translation();
  q[0] = 0.5;
  const auto p = forward(q).translation();
  EXPECT_NEAR((p.head<2>() - axis).norm(), (p0.head<2>() - axis).norm(), 1e-4);
  EXPECT_NEAR(p.z(), p0.z(), 1e-4);
  // The joint origin rpy (pi, 0, -pi) flips its z axis down: positive pan swings toward -y.
  EXPECT_LT(p.y(), 0.0);
}

TEST(Kinematics, JacobianShapeAndFinite) {
  JointVector q;
  q << 0.1, -0.3, 0.4, 1.2, 0.0;
  const auto J = numeric_jacobian(q);
  EXPECT_TRUE(J.allFinite());
  EXPECT_GT(J.topRows<3>().norm(), 0.0);
}

TEST(Kinematics, InverseIsTodo) {
  EXPECT_FALSE(inverse(Eigen::Vector3d(0.2, 0, 0.08), Eigen::Vector3d(0, 0, -1),
                       JointVector::Zero()).has_value());
}
