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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__EGO_UNCERTAINTY_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__EGO_UNCERTAINTY_HPP_

#include <Eigen/Core>

namespace autoware::multi_object_tracker
{

// Per-cycle ego (localization) uncertainty, expressed in the map/world frame, used to grow each
// tracked object's position process noise during prediction. The projection into a given object's
// world position (range r, bearing theta from the ego) is:
//   dQ_pos(dt, r) = (dt / tau) * [ pos_cov + yaw_var * r^2 * (t_hat t_hat^T) ]   (ego POSE error)
//                 +  dt^2      *   vel_cov                                       (ego MOTION)
// with t_hat the tangential unit vector (perpendicular to the ego->object line of sight). Both
// terms are functions of dt and vanish at dt = 0; the yaw-lever term also vanishes at r = 0 (as
// the ego reaches the object). tau is the localization-error correlation time over which the full
// ego pose covariance is leaked into the process noise. A default-constructed (zero) instance
// makes the injection a no-op, preserving legacy behavior when odometry uncertainty is disabled.
struct EgoUncertainty
{
  double ego_x{0.0};                                 // ego position in the map frame [m]
  double ego_y{0.0};                                 // ego position in the map frame [m]
  Eigen::Matrix2d pos_cov{Eigen::Matrix2d::Zero()};  // ego position covariance (map) [m^2]
  double yaw_var{0.0};                               // ego heading variance [rad^2]
  Eigen::Matrix2d vel_cov{Eigen::Matrix2d::Zero()};  // ego velocity covariance (map) [m^2/s^2]
  double inv_correlation_time{0.0};                  // 1 / tau [1/s]; <= 0 disables the pose term

  // Fast path: nothing to add when the pose term is disabled and there is no velocity covariance.
  bool isZero() const { return inv_correlation_time <= 0.0 && vel_cov.isZero(); }
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__MOTION_MODEL__EGO_UNCERTAINTY_HPP_
