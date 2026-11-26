#include "gambit/kinematics/so101.hpp"

#include <stdexcept>

namespace gambit::kinematics {

// so101_joints() is constant data (copied from the URDF), so it stays filled in.
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

Eigen::Matrix3d rpy_to_matrix(const Eigen::Vector3d& /*rpy*/) {
  // Design: URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) * Ry(pitch) * Rx(roll). With Eigen,
  // compose three Eigen::AngleAxisd (UnitZ, UnitY, UnitX) and call toRotationMatrix().
  // Test one nonzero angle at a time; the multiplication order is the usual mistake.
  throw std::logic_error("TODO: rpy_to_matrix");
}

Eigen::Isometry3d so101_tool_offset() {
  // Design: the fixed gripper_frame_joint from the URDF:
  //   xyz = (-0.0079, -0.000218121, -0.0981274), rpy = (0, 3.14159, 0)
  // (same numbers as HARDCODED_CHAIN in kinematics.ipynb). Build an Eigen::Isometry3d with
  // translation() = xyz and linear() = rpy_to_matrix(rpy).
  throw std::logic_error("TODO: so101_tool_offset");
}

Eigen::Isometry3d forward(const JointVector& /*q*/) {
  // Design:
  // - Start at Identity (arm base frame). For each joint i in so101_joints():
  //   T = T * Origin_i * Rot(axis_i, q_i), where Origin_i = Trans(xyz) * rpy_to_matrix(rpy)
  //   and Rot is Eigen::AngleAxisd(q_i, axis.normalized()).
  // - Finish with T * so101_tool_offset().
  // - Eigen::Isometry3d keeps the rotation part orthonormal-friendly and composes with *.
  // - Optimisation for later (measure first): cache the constant origin transforms.
  // Must reproduce the notebook's fk(zeros) tip (0.3914, 0.0, 0.2265).
  throw std::logic_error("TODO: forward");
}

Eigen::Matrix<double, 6, kNumJoints> numeric_jacobian(const JointVector& /*q*/, double /*eps*/) {
  // Design:
  // - Task vector v(q) = [tool translation; tool z axis (linear().col(2))] (6x1).
  // - Column j = (v(q + eps e_j) - v(q - eps e_j)) / (2 eps): central differences, 10 FKs.
  // - A small lambda for v(q) keeps the loop readable. An analytic Jacobian is a later
  //   optimisation if profiling says FK cost matters.
  throw std::logic_error("TODO: numeric_jacobian");
}

std::optional<IkResult> inverse(const Eigen::Vector3d& /*target_position*/,
                                const Eigen::Vector3d& /*approach*/,
                                const JointVector& /*seed*/) {
  // Design (port of ik_dls() from kinematics.ipynb once the Python one is validated):
  // - Iterate from the seed: e = [p* - p(q); w (a* - a(q))], J = W * numeric_jacobian(q).
  // - DLS step dq = J^T (J J^T + lambda^2 I)^-1 e; solve with Eigen's ldlt() rather than
  //   inverse() (symmetric positive definite, cheaper and more stable).
  // - Clamp to the JointSpec lower/upper after each step (or better limit handling).
  // - Converged when position and approach errors are under tolerance -> IkResult;
  //   no convergence within the iteration budget -> std::nullopt.
  throw std::logic_error("TODO: inverse");
}

}  // namespace gambit::kinematics
