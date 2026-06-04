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

#define EIGEN_MPL2_ONLY

#include "autoware/multi_object_tracker/tracker/model/pedestrian_tracker.hpp"

#include "autoware/multi_object_tracker/object_model/shapes.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autoware_utils_geometry/boost_polygon_utils.hpp>
#include <autoware_utils_math/normalization.hpp>
#include <autoware_utils_math/unit_conversion.hpp>
#include <tf2/utils.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <bits/stdc++.h>

namespace autoware::multi_object_tracker
{
PedestrianTracker::PedestrianTracker(const rclcpp::Time & time, const types::DynamicObject & object)
: Tracker(time, object), logger_(rclcpp::get_logger("PedestrianTracker"))
{
  tracker_type_ = TrackerType::PEDESTRIAN;

  using Shape = autoware_perception_msgs::msg::Shape;
  if (object.shape.type == Shape::POLYGON) {
    types::DynamicObject bbox_object;
    if (shapes::convertConvexHullToBoundingBox(object, bbox_object, std::nullopt)) {
      object_.shape.dimensions.x = bbox_object.shape.dimensions.x;
      object_.shape.dimensions.y = bbox_object.shape.dimensions.y;
      object_.pose.orientation = bbox_object.pose.orientation;
    } else {
      object_.shape.dimensions.x = object_model_.init_size.length;
      object_.shape.dimensions.y = object_model_.init_size.width;
      object_.shape.dimensions.z = object_model_.init_size.height;
    }
  } else if (object.shape.type == Shape::CYLINDER) {
    const double diameter = std::max(object_.shape.dimensions.x, object_.shape.dimensions.y);
    object_.shape.dimensions.x = diameter;
    object_.shape.dimensions.y = diameter;
  }
  object_.shape.type = Shape::BOUNDING_BOX;
  object_.shape.footprint.points.clear();
  // set maximum and minimum size
  // limitObjectExtension(object_model_);

  // Set motion model parameters
  {
    const double q_stddev_x = object_model_.process_noise.vel_long;
    const double q_stddev_y = object_model_.process_noise.vel_lat;
    const double q_stddev_yaw = object_model_.process_noise.yaw_rate;
    const double q_stddev_vx = object_model_.process_noise.acc_long;
    const double q_stddev_wz = object_model_.process_noise.acc_turn;
    motion_model_.setMotionParams(q_stddev_x, q_stddev_y, q_stddev_yaw, q_stddev_vx, q_stddev_wz);
  }

  // Set motion limits
  {
    const double max_vel = object_model_.process_limit.vel_long_max;
    const double max_turn_rate = object_model_.process_limit.yaw_rate_max;
    motion_model_.setMotionLimits(max_vel, max_turn_rate);  // maximum velocity and slip angle
  }

  // Set initial state
  {
    using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    const double x = object.pose.position.x;
    const double y = object.pose.position.y;
    const double yaw = tf2::getYaw(object.pose.orientation);

    auto pose_cov = object.pose_covariance;
    if (!object.kinematics.has_position_covariance) {
      // initial state covariance
      const auto & p0_cov_x = object_model_.initial_covariance.pos_x;
      const auto & p0_cov_y = object_model_.initial_covariance.pos_y;
      const auto & p0_cov_yaw = object_model_.initial_covariance.yaw;

      const double cos_yaw = std::cos(yaw);
      const double sin_yaw = std::sin(yaw);
      const double sin_2yaw = std::sin(2.0 * yaw);
      pose_cov[XYZRPY_COV_IDX::X_X] = p0_cov_x * cos_yaw * cos_yaw + p0_cov_y * sin_yaw * sin_yaw;
      pose_cov[XYZRPY_COV_IDX::X_Y] = 0.5 * (p0_cov_x - p0_cov_y) * sin_2yaw;
      pose_cov[XYZRPY_COV_IDX::Y_Y] = p0_cov_x * sin_yaw * sin_yaw + p0_cov_y * cos_yaw * cos_yaw;
      pose_cov[XYZRPY_COV_IDX::Y_X] = pose_cov[XYZRPY_COV_IDX::X_Y];
      pose_cov[XYZRPY_COV_IDX::YAW_YAW] = p0_cov_yaw;
    }

    double vel = 0.0;
    double wz = 0.0;
    if (object.kinematics.has_twist) {
      vel = object.twist.linear.x;
      wz = object.twist.angular.z;
    }

    double vel_cov = object_model_.initial_covariance.vel_long;
    double wz_cov = object_model_.initial_covariance.yaw_rate;
    if (object.kinematics.has_twist_covariance) {
      vel_cov = object.twist_covariance[XYZRPY_COV_IDX::X_X];
      wz_cov = object.twist_covariance[XYZRPY_COV_IDX::YAW_YAW];
    }

    // initialize motion model
    motion_model_.initialize(time, x, y, yaw, pose_cov, vel, vel_cov, wz, wz_cov);
  }
}

bool PedestrianTracker::predict(const rclcpp::Time & time)
{
  return motion_model_.predictState(time);
}

bool PedestrianTracker::measureWithPose(const types::DynamicObject & object)
{
  // update motion model
  bool is_updated = false;
  {
    const double x = object.pose.position.x;
    const double y = object.pose.position.y;

    is_updated = motion_model_.updateStatePose(x, y, object.pose_covariance);
    motion_model_.limitStates();
  }

  // position z
  constexpr double gain = 0.1;
  object_.pose.position.z = (1.0 - gain) * object_.pose.position.z + gain * object.pose.position.z;

  // remove cached object
  removeCache();

  return is_updated;
}

bool PedestrianTracker::measureWithShape(
  const types::DynamicObject & object, const rclcpp::Time & time, const bool trust_extension)
{
  using Shape = autoware_perception_msgs::msg::Shape;
  // const auto & sl = object_model_.size_limit;
  // const double len_max = sl.length_max * 1.5;
  // const double len_min = sl.length_min * 0.5;
  // const double wid_max = sl.width_max * 1.5;
  // const double wid_min = sl.width_min * 0.5;
  // const double hgt_max = sl.height_max * 1.5;
  // const double hgt_min = sl.height_min * 0.5;

  if (object.shape.type == Shape::BOUNDING_BOX) {
    const double dim_x = object.shape.dimensions.x;
    const double dim_y = object.shape.dimensions.y;
    const double dim_z = object.shape.dimensions.z;
    // if (
    //   dim_x > len_max || dim_x < len_min || dim_y > wid_max || dim_y < wid_min ||
    //   dim_z > hgt_max || dim_z < hgt_min) {
    //   return false;
    // }
    const double gain = trust_extension ? 0.5 : 0.0;
    const double gain_inv = 1.0 - gain;
    object_.shape.dimensions.x = gain_inv * object_.shape.dimensions.x + gain * dim_x;
    object_.shape.dimensions.y = gain_inv * object_.shape.dimensions.y + gain * dim_y;
    object_.shape.dimensions.z = gain_inv * object_.shape.dimensions.z + gain * dim_z;
  } else if (object.shape.type == Shape::CYLINDER) {
    const double diameter = std::max(object.shape.dimensions.x, object.shape.dimensions.y);
    const double dim_z = object.shape.dimensions.z;
    // if (diameter > wid_max || diameter < wid_min || dim_z > hgt_max || dim_z < hgt_min) {
    //   return false;
    // }
    const double gain = trust_extension ? 0.5 : 0.0;
    const double gain_inv = 1.0 - gain;
    object_.shape.dimensions.x = gain_inv * object_.shape.dimensions.x + gain * diameter;
    object_.shape.dimensions.y = gain_inv * object_.shape.dimensions.y + gain * diameter;
    object_.shape.dimensions.z = gain_inv * object_.shape.dimensions.z + gain * dim_z;
  } else {
    // POLYGON: project cluster footprint onto tracker heading
    geometry_msgs::msg::Pose current_pose;
    std::array<double, 36> pose_cov;
    geometry_msgs::msg::Twist current_twist;
    std::array<double, 36> twist_cov;
    if (!motion_model_.getPredictedState(time, current_pose, pose_cov, current_twist, twist_cov)) {
      return false;
    }
    const double tracker_yaw = tf2::getYaw(current_pose.orientation);
    const auto aligned = shapes::alignClusterToOrientation(object, tracker_yaw);
    if (!aligned) {
      return false;
    }
    const double dim_x = aligned->shape.dimensions.x;
    const double dim_y = aligned->shape.dimensions.y;
    const double dim_z = aligned->shape.dimensions.z;
    // if (
    //   dim_x > len_max || dim_x < len_min || dim_y > wid_max || dim_y < wid_min ||
    //   dim_z > hgt_max || dim_z < hgt_min) {
    //   return false;
    // }
    // footprint is direct geometry — use low gain even when !trust_extension
    const double gain = trust_extension ? 0.5 : 0.4;
    const double gain_inv = 1.0 - gain;
    object_.shape.dimensions.x = gain_inv * object_.shape.dimensions.x + gain * dim_x;
    object_.shape.dimensions.y = gain_inv * object_.shape.dimensions.y + gain * dim_y;
    object_.shape.dimensions.z = gain_inv * object_.shape.dimensions.z + gain * dim_z;

    object_.shape.footprint= object.shape.footprint;
  }

  object_.shape.type = Shape::BOUNDING_BOX;
  object_.shape.footprint.points.clear();
  object_.area = types::getArea(object_.shape);
  // limitObjectExtension(object_model_);
  return true;
}

bool PedestrianTracker::measure(
  const types::DynamicObject & object, const rclcpp::Time & time,
  const types::InputChannel & channel_info)
{
  // check time gap
  const double dt = motion_model_.getDeltaTime(time);
  if (0.01 /*10msec*/ < dt) {
    RCLCPP_WARN(
      logger_,
      "PedestrianTracker::measure There is a large gap between predicted time and measurement "
      "time. (%f)",
      dt);
  }

  // update object
  measureWithPose(object);
  measureWithShape(object, time, channel_info.trust_extension);

  return true;
}

bool PedestrianTracker::getTrackedObject(
  const rclcpp::Time & time, types::DynamicObject & object, const bool to_publish) const
{
  // try to return cached object
  if (!getCachedObject(time, object)) {
    // if there is no cached object, predict and update cache
    object = object_;
    object.time = time;

    // predict from motion model
    auto & pose = object.pose;
    auto & pose_cov = object.pose_covariance;
    auto & twist = object.twist;
    auto & twist_cov = object.twist_covariance;
    if (!motion_model_.getPredictedState(time, pose, pose_cov, twist, twist_cov)) {
      RCLCPP_WARN(logger_, "PedestrianTracker::getTrackedObject: Failed to get predicted state.");
      return false;
    }

    // cache object
    updateCache(object, time);
  }

  // if the tracker is to be published, check twist uncertainty
  // in case the twist uncertainty is large, lower the twist value
  if (to_publish) {
    using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
    // lower the x twist magnitude 1 sigma smaller
    // if the twist is smaller than 1 sigma, the twist is zeroed
    auto & twist = object.twist;
    constexpr double vel_cov_buffer = 0.7;  // [m/s] buffer not to limit certain twist
    constexpr double vel_too_low_ignore =
      0.35;  // [m/s] if the velocity is lower than this, do not limit
    const double vel_long = std::abs(twist.linear.x);
    if (vel_long > vel_too_low_ignore) {
      const double vel_limit = std::max(
        std::sqrt(object.twist_covariance[XYZRPY_COV_IDX::X_X]) - vel_cov_buffer, 0.0);  // [m/s]

      if (vel_long < vel_limit) {
        twist.linear.x = twist.linear.x > 0 ? vel_too_low_ignore : -vel_too_low_ignore;
      } else {
        twist.linear.x =
          twist.linear.x > 0 ? twist.linear.x - vel_limit : twist.linear.x + vel_limit;
      }
    }
  }

  return true;
}

}  // namespace autoware::multi_object_tracker
