#include "gambit/kinematics/so101.hpp"

namespace gambit::kinematics {

const std::array<JointSpec, kNumJoints>& so101_joints() {
  // Source: gambit_description/urdf/so101.urdf.xacro (keep in sync if the URDF changes).
  static const std::array<JointSpec, kNumJoints> joints{{
      {"shoulder_pan", {0.0388353, -8.97657e-09, 0.0624}, {3.14159, 4.18253e-17, -3.14159},
       {0, 0, 1}, -1.91986, 1.91986},
      {"shoulder_lift", {-0.0303992, -0.0182778, -0.0542}, {-1.5708, -1.5708, 0}, {0, 0, 1},
       -1.74533, 1.74533},
      {"elbow_flex", {-0.11257, -0.028, 1.73763e-16}, {-3.63608e-16, 8.74301e-16, 1.5708},
       {0, 0, 1}, -1.69, 1.69},
      {"wrist_flex", {-0.1349, 0.0052, 3.62355e-17}, {4.02456e-15, 8.67362e-16, -1.5708},
       {0, 0, 1}, -1.65806, 1.65806},
      {"wrist_roll", {5.55112e-17, -0.0611, 0.0181}, {1.5708, 0.0486795, 3.14159}, {0, 0, 1},
       -2.74385, 2.84121},
  }};
  return joints;
}

Eigen::Matrix3d rpy_to_matrix(const Eigen::Vector3d& rpy) {
  return (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

Eigen::Isometry3d so101_tool_offset() {
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() = Eigen::Vector3d(-0.0079, -0.000218121, -0.0981274);
  t.linear() = rpy_to_matrix(Eigen::Vector3d(0, 3.14159, 0));
  return t;
}

Eigen::Isometry3d forward(const JointVector& q) {
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  const auto& joints = so101_joints();
  for (int i = 0; i < kNumJoints; ++i) {
    Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();
    origin.translation() = joints[i].xyz;
    origin.linear() = rpy_to_matrix(joints[i].rpy);
    Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
    motion.linear() = Eigen::AngleAxisd(q[i], joints[i].axis.normalized()).toRotationMatrix();
    t = t * origin * motion;
  }
  return t * so101_tool_offset();
}

Eigen::Matrix<double, 6, kNumJoints> numeric_jacobian(const JointVector& q, double eps) {
  Eigen::Matrix<double, 6, kNumJoints> J;
  auto task = [](const JointVector& qq) {
    const Eigen::Isometry3d t = forward(qq);
    Eigen::Matrix<double, 6, 1> v;
    v << t.translation(), t.linear().col(2);
    return v;
  };
  for (int j = 0; j < kNumJoints; ++j) {
    JointVector dq = JointVector::Zero();
    dq[j] = eps;
    J.col(j) = (task(q + dq) - task(q - dq)) / (2 * eps);
  }
  return J;
}

std::optional<IkResult> inverse(const Eigen::Vector3d& /*target_position*/,
                                const Eigen::Vector3d& /*approach*/,
                                const JointVector& /*seed*/) {
  // TODO: port ik_dls() from kinematics.ipynb:
  //   e = [p* - p(q); w * (a* - a(q))],  dq = J^T (J J^T + lambda^2 I)^-1 e,  q = clamp(q + dq)
  // Add joint-limit handling and a convergence test, then enable the IK round-trip test.
  return std::nullopt;
}

}  // namespace gambit::kinematics
