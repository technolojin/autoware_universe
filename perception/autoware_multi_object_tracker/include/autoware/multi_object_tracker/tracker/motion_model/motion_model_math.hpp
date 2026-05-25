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

#include <tf2/LinearMath/Quaternion.hpp>

#include <geometry_msgs/msg/pose.hpp>

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
  const double cov_long, const double cov_lat,
  const double sin_yaw_sq, const double cos_yaw_sq, const double sin_cos_yaw)
{
  return {
    cov_long * cos_yaw_sq + cov_lat * sin_yaw_sq,  // xx
    (cov_long - cov_lat) * sin_cos_yaw,             // xy = yx
    cov_long * sin_yaw_sq + cov_lat * cos_yaw_sq    // yy
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
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
}

}  // namespace autoware::multi_object_tracker::motion_model_math

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__MOTION_MODEL_MATH_HPP_
