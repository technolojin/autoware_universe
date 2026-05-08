// Copyright 2020 TIER IV, Inc.
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

#include "autoware/multi_object_tracker/tracker/model/multiple_vehicle_tracker.hpp"

#include "autoware/multi_object_tracker/object_model/object_model.hpp"

#include <tf2/utils.hpp>

namespace autoware::multi_object_tracker
{

MultipleVehicleTracker::MultipleVehicleTracker(
  const rclcpp::Time & time, const types::DynamicObject & object)
: VehicleTracker(object_model::normal_vehicle, time, object),
  big_object_model_(object_model::big_vehicle),
  big_shape_update_anchor_(BicycleMotionModel::LengthUpdateAnchor::CENTER)
{
  tracker_type_ = TrackerType::MULTIPLE_VEHICLE;

  big_motion_model_.setMotionParams(
    big_object_model_.process_noise, big_object_model_.bicycle_state,
    big_object_model_.process_limit);

  // Initialize big model with the same initial state as the normal model
  using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  const double x = object.pose.position.x;
  const double y = object.pose.position.y;
  const double yaw = tf2::getYaw(object.pose.orientation);

  auto pose_cov = object.pose_covariance;
  if (!object.kinematics.has_position_covariance) {
    const auto & p0_cov_x = big_object_model_.initial_covariance.pos_x;
    const auto & p0_cov_y = big_object_model_.initial_covariance.pos_y;
    const auto & p0_cov_yaw = big_object_model_.initial_covariance.yaw;
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double sin_2yaw = std::sin(2.0 * yaw);
    pose_cov[XYZRPY_COV_IDX::X_X] = p0_cov_x * cos_yaw * cos_yaw + p0_cov_y * sin_yaw * sin_yaw;
    pose_cov[XYZRPY_COV_IDX::X_Y] = 0.5 * (p0_cov_x - p0_cov_y) * sin_2yaw;
    pose_cov[XYZRPY_COV_IDX::Y_Y] = p0_cov_x * sin_yaw * sin_yaw + p0_cov_y * cos_yaw * cos_yaw;
    pose_cov[XYZRPY_COV_IDX::Y_X] = pose_cov[XYZRPY_COV_IDX::X_Y];
    pose_cov[XYZRPY_COV_IDX::YAW_YAW] = p0_cov_yaw;
  }

  double vel_x = 0.0;
  double vel_y = 0.0;
  double vel_x_cov = big_object_model_.initial_covariance.vel_long;
  double vel_y_cov = big_object_model_.bicycle_state.init_slip_angle_cov;
  if (object.kinematics.has_twist) {
    vel_x = object.twist.linear.x;
    vel_y = object.twist.linear.y;
  }
  if (object.kinematics.has_twist_covariance) {
    vel_x_cov = object.twist_covariance[XYZRPY_COV_IDX::X_X];
    vel_y_cov = object.twist_covariance[XYZRPY_COV_IDX::Y_Y];
  }

  const double & length = object_.shape.dimensions.x;
  big_motion_model_.initialize(time, x, y, yaw, pose_cov, vel_x, vel_x_cov, vel_y, vel_y_cov, length);
}

bool MultipleVehicleTracker::predict(const rclcpp::Time & time)
{
  VehicleTracker::predict(time);
  big_motion_model_.predictState(time);
  return true;
}

bool MultipleVehicleTracker::measure(
  const types::DynamicObject & in_object, const rclcpp::Time & time,
  const types::InputChannel & channel_info)
{
  VehicleTracker::measure(in_object, time, channel_info);
  measureKinematics(in_object, channel_info, big_motion_model_);
  return true;
}

bool MultipleVehicleTracker::conditionedUpdate(
  const types::DynamicObject & measurement, const types::DynamicObject & prediction,
  const autoware_perception_msgs::msg::Shape & tracker_shape, const rclcpp::Time & measurement_time,
  const types::InputChannel & channel_info)
{
  // Determine strategy once — pure computation, same result for both models
  UpdateStrategy strategy = determineUpdateStrategy(measurement, prediction);

  if (strategy.type == UpdateStrategyType::WEAK_UPDATE) {
    // WEAK_UPDATE calls measure() (virtual → MultipleVehicleTracker::measure) → both models updated
    types::DynamicObject pseudo_measurement = prediction;
    createPseudoMeasurement(measurement, pseudo_measurement, tracker_shape, true);
    measure(pseudo_measurement, measurement_time, channel_info);
    return true;
  }

  // FRONT or REAR wheel update: apply to both models with the same strategy
  const std::array<double, 36> pose_cov = measurement.pose_covariance;
  bool is_updated =
    applyConditionedUpdate(strategy, pose_cov, motion_model_, shape_update_anchor_);
  is_updated = is_updated & applyConditionedUpdate(strategy, pose_cov, big_motion_model_, big_shape_update_anchor_);
  removeCache();
  return is_updated;
}

void MultipleVehicleTracker::setObjectShape(const autoware_perception_msgs::msg::Shape & shape)
{
  VehicleTracker::setObjectShape(shape);
  if (shape.type == autoware_perception_msgs::msg::Shape::BOUNDING_BOX) {
    updateModelLength(shape.dimensions.x, big_motion_model_, big_shape_update_anchor_);
  }
}

bool MultipleVehicleTracker::getTrackedObject(
  const rclcpp::Time & time, types::DynamicObject & object, const bool to_publish) const
{
  const auto label = getHighestProbLabel();

  if (
    label == classes::Label::BUS || label == classes::Label::TRUCK ||
    label == classes::Label::TRAILER) {
    // Get base state (z position, shape, cache) from normal model without velocity suppression
    VehicleTracker::getTrackedObject(time, object, false);

    // Override kinematics with big model's predicted state.
    // Pass object.pose directly so getPredictedState updates x/y/orientation in-place
    // while leaving pose.position.z intact (bicycle model has no z state).
    if (!big_motion_model_.getPredictedState(
          time, object.pose, object.pose_covariance, object.twist, object.twist_covariance)) {
      return false;
    }
    object.shape.dimensions.x = big_motion_model_.getLength();

    if (to_publish) {
      applyVelocitySuppression(object);
    }
  } else {
    VehicleTracker::getTrackedObject(time, object, to_publish);
  }

  return true;
}

}  // namespace autoware::multi_object_tracker
