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

#include "autoware/multi_object_tracker/tracker/update/vehicle_update_strategy.hpp"

#include <autoware_utils_geometry/msg/covariance.hpp>
#include <autoware_utils_math/normalization.hpp>
#include <tf2/utils.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace autoware::multi_object_tracker
{

namespace
{

constexpr double ALIGNMENT_RATIO_THRESHOLD = 0.2;     // 20% of the larger object's length
constexpr double ALIGNMENT_ABSOLUTE_THRESHOLD = 1.0;  // [m] minimum tolerance for small objects
constexpr double TRUST_MIN = 0.3;  // below this the corner geometry is too weak -> weak update

}  // namespace

UpdateStrategy determineUpdateStrategy(
  const shapes::PolygonGeometry & geometry, const types::DynamicObject & prediction,
  double tracked_width)
{
  UpdateStrategy strategy;
  strategy.type = UpdateStrategyType::WEAK_UPDATE;

  const double pred_yaw = tf2::getYaw(prediction.pose.orientation);
  // Front vs rear from the longitudinal projection onto the predicted body axis.
  const double ux = std::cos(pred_yaw), uy = std::sin(pred_yaw);
  const double center_proj = prediction.pose.position.x * ux + prediction.pose.position.y * uy;
  const double half_len = 0.5 * prediction.shape.dimensions.x;

  // Classify a reconstructed end-face center into a front/rear wheel update, rejecting (-> weak) a
  // face that is grossly inconsistent with a predicted end face.
  auto finalizeFace = [&](const geometry_msgs::msg::Point & face) {
    const double face_proj = face.x * ux + face.y * uy;
    const double dist_to_face = std::abs(std::abs(face_proj - center_proj) - half_len);
    const double max_len = std::max(prediction.shape.dimensions.x, geometry.long_edge_len);
    const double alignment_threshold =
      std::max(ALIGNMENT_RATIO_THRESHOLD * max_len, ALIGNMENT_ABSOLUTE_THRESHOLD);
    if (dist_to_face > alignment_threshold) {
      strategy.type = UpdateStrategyType::WEAK_UPDATE;
      return;
    }
    strategy.type = (face_proj >= center_proj) ? UpdateStrategyType::FRONT_WHEEL_UPDATE
                                               : UpdateStrategyType::REAR_WHEEL_UPDATE;
    strategy.anchor_point = face;
  };

  if (geometry.has_corner && geometry.trust >= TRUST_MIN) {
    // Body axis used to place the anchor: the observed long edge when trustworthy (lets the EKF
    // rotate toward it via the wheel lever), otherwise the predicted axis (yaw held).
    const double theta = geometry.yaw_cue_valid ? geometry.long_edge_dir : pred_yaw;
    const double nx = -std::sin(theta);  // body lateral unit vector
    const double ny = std::cos(theta);

    // Reconstruct the observed end-face CENTER from the near corner: step inward (toward the
    // tracked center) by half the tracked width along the lateral axis. Width comes from the
    // tracker, never from the (possibly inflated / partial) polygon extent.
    const double to_cx = prediction.pose.position.x - geometry.near_corner.x;
    const double to_cy = prediction.pose.position.y - geometry.near_corner.y;
    const double sgn = (nx * to_cx + ny * to_cy) >= 0.0 ? 1.0 : -1.0;
    geometry_msgs::msg::Point face;
    face.x = geometry.near_corner.x + sgn * (tracked_width * 0.5) * nx;
    face.y = geometry.near_corner.y + sgn * (tracked_width * 0.5) * ny;
    face.z = geometry.near_corner.z;
    finalizeFace(face);
    return strategy;
  }

  if (geometry.has_end_face && geometry.trust >= TRUST_MIN) {
    // Single visible end face (thin rear/front cluster): the observed edge midpoint IS the end-face
    // center, so anchor there directly. Partial-width / front-rear uncertainty is absorbed
    // downstream by correctWheelAnchorLateral (using observed_width) and the gross-mismatch gate.
    finalizeFace(geometry.end_face_center);
    return strategy;
  }

  return strategy;  // WEAK_UPDATE: no usable corner or end-face cue
}

void createPseudoMeasurement(
  const types::DynamicObject & meas, types::DynamicObject & pred,
  const autoware_perception_msgs::msg::Shape & tracker_shape, const bool enlarge_covariance)
{
  // Apply linear fall‑off weight on dist square
  const double dx = meas.pose.position.x - pred.pose.position.x;
  const double dy = meas.pose.position.y - pred.pose.position.y;
  const double dist2 = dx * dx + dy * dy;
  constexpr double d_max_square_inv = 1 / 2.0;  // saturate when distance overs 1.414 m
  constexpr double min_w = 0.0;
  const double w_pose = std::clamp(1.0 - dist2 * d_max_square_inv, min_w, 1.0);

  // Blend position (x, y, z)
  pred.pose.position.x = pred.pose.position.x * (1 - w_pose) + meas.pose.position.x * w_pose;
  pred.pose.position.y = pred.pose.position.y * (1 - w_pose) + meas.pose.position.y * w_pose;
  pred.pose.position.z = pred.pose.position.z * (1 - w_pose) + meas.pose.position.z * w_pose;

  // Use smoothed shape and its area
  pred.shape = tracker_shape;
  pred.area = types::getArea(tracker_shape);

  // Blend orientation
  if (meas.kinematics.orientation_availability != types::OrientationAvailability::UNAVAILABLE) {
    double yaw_pred = tf2::getYaw(pred.pose.orientation);
    double yaw_meas = tf2::getYaw(meas.pose.orientation);

    double yaw_diff = autoware_utils_math::normalize_radian(yaw_meas - yaw_pred);
    // Handle SIGN_UNKNOWN: limit yaw difference to [-90°, 90°]
    if (meas.kinematics.orientation_availability == types::OrientationAvailability::SIGN_UNKNOWN) {
      if (yaw_diff > M_PI_2) {
        yaw_diff -= M_PI;
      } else if (yaw_diff < -M_PI_2) {
        yaw_diff += M_PI;
      }
    }
    double yaw_fused = yaw_pred + yaw_diff * w_pose;
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw_fused);
    pred.pose.orientation = tf2::toMsg(q);
  }

  // Enlarge covariance if requested (for weak updates)
  if (enlarge_covariance) {
    using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    constexpr double additional_position_cov = 9.0;     // [m^2] additional variance
    constexpr double additional_orientation_cov = 0.5;  // [rad^2] additional variance
    constexpr double additional_velocity_cov = 25.0;    // [m^2/s^2] additional variance

    pred.pose_covariance[XYZRPY_COV_IDX::X_X] += additional_position_cov;
    pred.pose_covariance[XYZRPY_COV_IDX::Y_Y] += additional_position_cov;
    pred.pose_covariance[XYZRPY_COV_IDX::YAW_YAW] += additional_orientation_cov;

    // Enlarge velocity covariance if available
    if (pred.kinematics.has_twist_covariance) {
      pred.twist_covariance[XYZRPY_COV_IDX::X_X] += additional_velocity_cov;
      pred.twist_covariance[XYZRPY_COV_IDX::Y_Y] += additional_velocity_cov;
    }
  }
}

WheelAnchorLateral correctWheelAnchorLateral(
  double yaw, double tracker_width, const geometry_msgs::msg::Point & tracker_center,
  double polygon_width, const geometry_msgs::msg::Point & anchor, double balance_alpha,
  double corner_residual_beta)
{
  WheelAnchorLateral result{anchor, 0.0};

  const double slack = 0.5 * (polygon_width - tracker_width);
  if (slack <= 0.0) {
    // Polygon narrower than (or equal to) the tracker: partial view. Keep the anchor and add the
    // worst-case lateral offset (half the missing width) as variance.
    const double half_gap = -slack;  // = 0.5 * (tracker_width - polygon_width) >= 0
    result.var_lat = half_gap * half_gap;
    return result;
  }

  // Polygon wider than the tracker: soft dead-zone ("back-lash") lateral correction.
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  // Signed lateral offset of the observed edge center from the tracker center, along the body
  // lateral axis n = (-sin yaw, cos yaw). The longitudinal component is orthogonal to n and drops.
  const double d =
    -(anchor.x - tracker_center.x) * sin_yaw + (anchor.y - tracker_center.y) * cos_yaw;
  const double ad = std::abs(d);
  const double sgn = (d >= 0.0) ? 1.0 : -1.0;

  // Soft dead-zone: slope `balance_alpha` while contained (|d| <= slack), unit slope once the
  // corner is exposed. Continuous at |d| = slack.
  const double shift = (ad <= slack) ? balance_alpha * ad : (ad - slack) + balance_alpha * slack;
  const double lateral_move = sgn * shift - d;  // (corrected - observed) lateral offset, along n
  result.anchor.x = anchor.x - lateral_move * sin_yaw;
  result.anchor.y = anchor.y + lateral_move * cos_yaw;

  // Added lateral std: `slack` when centered (true position unknown across the slack), shrinking to
  // `corner_residual_beta` * slack once the corner is matched. Continuous in |d|.
  const double t = std::clamp(ad / slack, 0.0, 1.0);
  const double std_lat = slack * (1.0 - (1.0 - corner_residual_beta) * t);
  result.var_lat = std_lat * std_lat;
  return result;
}

}  // namespace autoware::multi_object_tracker
