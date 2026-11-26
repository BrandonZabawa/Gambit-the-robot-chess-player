// SO-101 forward kinematics (Eigen). Mirrors gambit_arm/notebooks/kinematics.ipynb.
// Joint origins/axes copied from gambit_description/urdf/so101.urdf.xacro
// (joints shoulder_pan .. wrist_roll, then the fixed gripper_frame_joint as the tool frame).
// URDF convention: T_parent_child = Trans(xyz) * Rz(yaw) * Ry(pitch) * Rx(roll) * Rot(axis, q).
#pragma once

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <string_view>

namespace gambit::kinematics {

inline constexpr int kNumJoints = 5;  // arm joints; the gripper is separate
using JointVector = Eigen::Matrix<double, kNumJoints, 1>;

struct JointSpec {
  std::string_view name;
  Eigen::Vector3d xyz;
  Eigen::Vector3d rpy;
  Eigen::Vector3d axis;
  double lower;
  double upper;
};

const std::array<JointSpec, kNumJoints>& so101_joints();
Eigen::Isometry3d so101_tool_offset();  // gripper_frame_joint (fixed)

Eigen::Matrix3d rpy_to_matrix(const Eigen::Vector3d& rpy);

// Pose of the tool frame (gripper_frame_link) in the arm base frame.
Eigen::Isometry3d forward(const JointVector& q);

// 6x5 numeric Jacobian of [position; approach (tool z axis)] w.r.t. q (central differences).
Eigen::Matrix<double, 6, kNumJoints> numeric_jacobian(const JointVector& q, double eps = 1e-6);

struct IkResult {
  JointVector q;
  double position_error;
  bool converged;
};

// TODO: damped-least-squares IK for position + approach direction, matching the notebook's
// ik_dls(). Currently returns std::nullopt. Port it once the Python version is validated.
std::optional<IkResult> inverse(const Eigen::Vector3d& target_position,
                                const Eigen::Vector3d& approach, const JointVector& seed);

}  // namespace gambit::kinematics
