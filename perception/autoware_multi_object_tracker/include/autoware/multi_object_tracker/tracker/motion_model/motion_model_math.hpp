// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__MOTION_MODEL_MATH_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__MOTION_MODEL_MATH_HPP_

#include <geometry_msgs/msg/pose.hpp>

#include <Eigen/Geometry>

#include <cmath>

namespace autoware::multi_object_tracker::motion_model_math
{

// Tentative default covariance for axes that are not modelled (Z, ROLL, PITCH, etc.)
constexpr double kUnobservedCov = 0.1 * 0.1;  // [m^2 or rad^2]

// Result of rotating a diagonal 2D covariance into the world frame.
struct RotatedCov2D
{
  double xx, xy, yy;
};

// Rotate diagonal process-noise covariance (cov_long, cov_lat) by yaw.
// Accepts pre-computed trigonometric products so no extra trig calls are introduced.
// Formula: R * diag(cov_long, cov_lat) * R^T  where R = Rot(yaw)
inline RotatedCov2D rotateDiagCov2D(
  const double cov_long, const double cov_lat, const double sin_yaw_sq, const double cos_yaw_sq,
  const double sin_cos_yaw)
{
  return {
    cov_long * cos_yaw_sq + cov_lat * sin_yaw_sq,  // xx
    (cov_long - cov_lat) * sin_cos_yaw,            // xy = yx
    cov_long * sin_yaw_sq + cov_lat * cos_yaw_sq   // yy
  };
}

// Clamp val symmetrically to [-abs_max, abs_max].
inline double clampSymmetric(const double val, const double abs_max)
{
  if (val > abs_max) return abs_max;
  if (val < -abs_max) return -abs_max;
  return val;
}

// Set pose orientation from a 2D yaw angle (roll = pitch = 0).
inline void setOrientationFromYaw(geometry_msgs::msg::Pose & pose, const double yaw)
{
  pose.orientation.x = 0.0;
  pose.orientation.y = 0.0;
  pose.orientation.z = std::sin(yaw * 0.5);
  pose.orientation.w = std::cos(yaw * 0.5);
}

// Rotate a full (non-diagonal) 2×2 covariance into the body frame:
// R(-yaw) * M * R(-yaw)^T
inline Eigen::Matrix2d rotateCov2D(const Eigen::Matrix2d & cov, const double yaw)
{
  const Eigen::Matrix2d R = Eigen::Rotation2Dd(-yaw).toRotationMatrix();
  return R * cov * R.transpose();
}

// Adjust measured_yaw to be continuous with estimated_yaw across π-wrap boundaries.
// Prevents sign flips in the Kalman innovation when yaw is observed directly.
inline double fixYawContinuity(const double estimated_yaw, const double measured_yaw)
{
  return measured_yaw + M_PI * std::round((estimated_yaw - measured_yaw) / M_PI);
}

}  // namespace autoware::multi_object_tracker::motion_model_math

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__MOTION_MODEL_MATH_HPP_
