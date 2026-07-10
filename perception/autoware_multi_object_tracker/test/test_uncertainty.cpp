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

// Tests for uncertainty::addOdometryUncertainty(): the choke point that folds the ego-pose
// (localization) error into each detection's measurement covariance R. The ego-heading term is
// what stops a strong bicycle motion model from turning an unmodeled ego-heading error into a
// persistent wrong object yaw, so these tests pin down where and how much that heading error
// inflates the object covariance.

#include "autoware/multi_object_tracker/types.hpp"
#include "autoware/multi_object_tracker/uncertainty/uncertainty_processor.hpp"

#include <autoware_utils_geometry/msg/covariance.hpp>
#include <rclcpp/time.hpp>

#include <nav_msgs/msg/odometry.hpp>

#include <gtest/gtest.h>

namespace
{
using autoware::multi_object_tracker::types::DynamicObject;
using autoware::multi_object_tracker::types::DynamicObjectList;
using autoware::multi_object_tracker::uncertainty::addOdometryUncertainty;
using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;

// Ego at the map origin, heading along +x (identity orientation), with a diagonal pose covariance.
// Twist and twist covariance are left zero so only the pose-error terms are exercised.
nav_msgs::msg::Odometry makeEgoOdometry(
  double pos_var_x, double pos_var_y, double yaw_var, const rclcpp::Time & stamp)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.pose.pose.orientation.w = 1.0;  // ego yaw = 0
  odom.pose.covariance.fill(0.0);
  odom.pose.covariance[XYZRPY_COV_IDX::X_X] = pos_var_x;
  odom.pose.covariance[XYZRPY_COV_IDX::Y_Y] = pos_var_y;
  odom.pose.covariance[XYZRPY_COV_IDX::YAW_YAW] = yaw_var;
  odom.twist.covariance.fill(0.0);
  return odom;
}

// A single object at (x, y) in the map frame, heading along +x, with a tiny diagonal covariance.
DynamicObjectList makeSingleObject(double x, double y, const rclcpp::Time & stamp)
{
  DynamicObject obj;
  obj.pose.position.x = x;
  obj.pose.position.y = y;
  obj.pose.orientation.w = 1.0;  // object yaw = 0
  obj.pose_covariance.fill(0.0);
  obj.pose_covariance[XYZRPY_COV_IDX::X_X] = 1e-3;
  obj.pose_covariance[XYZRPY_COV_IDX::Y_Y] = 1e-3;
  obj.pose_covariance[XYZRPY_COV_IDX::YAW_YAW] = 1e-4;
  obj.twist_covariance.fill(0.0);

  DynamicObjectList list;
  list.header.stamp = stamp;
  list.objects.push_back(obj);
  return list;
}

constexpr double kInitPosVar = 1e-3;
constexpr double kInitYawVar = 1e-4;
}  // namespace

// Ego-heading error inflates the object covariance perpendicular to the line of sight (lateral for
// an object straight ahead) proportionally to the squared range, and adds directly to the object
// yaw variance. The longitudinal (line-of-sight) component is untouched.
TEST(AddOdometryUncertainty, YawErrorInflatesLateralPositionAndYaw)
{
  const rclcpp::Time stamp(100, 0, RCL_ROS_TIME);
  const double range = 10.0;
  const double yaw_stddev = 0.017;  // ~1 deg
  const double yaw_var = yaw_stddev * yaw_stddev;

  auto objects = makeSingleObject(range, 0.0, stamp);  // straight ahead of the ego
  const auto odom = makeEgoOdometry(0.0, 0.0, yaw_var, stamp);
  addOdometryUncertainty(odom, objects);

  const auto & cov = objects.objects.front().pose_covariance;
  // Lateral (y) grows by yaw_var * range^2; longitudinal (x) is unchanged; yaw grows by yaw_var.
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::Y_Y], kInitPosVar + yaw_var * range * range, 1e-9);
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::X_X], kInitPosVar, 1e-9);
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::YAW_YAW], kInitYawVar + yaw_var, 1e-12);
}

// The heading-error inflation is directed perpendicular to the range vector: for an object off to
// the side the growth lands on the longitudinal (x) axis instead of the lateral (y) one.
TEST(AddOdometryUncertainty, YawErrorDirectionFollowsLineOfSight)
{
  const rclcpp::Time stamp(100, 0, RCL_ROS_TIME);
  const double range = 10.0;
  const double yaw_var = 0.017 * 0.017;

  auto objects = makeSingleObject(0.0, range, stamp);  // directly to the ego's left
  const auto odom = makeEgoOdometry(0.0, 0.0, yaw_var, stamp);
  addOdometryUncertainty(odom, objects);

  const auto & cov = objects.objects.front().pose_covariance;
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::X_X], kInitPosVar + yaw_var * range * range, 1e-9);
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::Y_Y], kInitPosVar, 1e-9);
}

// A larger ego-heading stddev must produce a strictly larger lateral inflation, scaling with the
// variance (stddev^2). This is the knob localization_error.yaw_stddev turns.
TEST(AddOdometryUncertainty, LateralInflationScalesWithYawVariance)
{
  const rclcpp::Time stamp(100, 0, RCL_ROS_TIME);
  const double range = 10.0;
  const double yaw_var_small = 0.01 * 0.01;
  const double yaw_var_large = 0.02 * 0.02;

  auto objects_small = makeSingleObject(range, 0.0, stamp);
  auto objects_large = makeSingleObject(range, 0.0, stamp);
  addOdometryUncertainty(makeEgoOdometry(0.0, 0.0, yaw_var_small, stamp), objects_small);
  addOdometryUncertainty(makeEgoOdometry(0.0, 0.0, yaw_var_large, stamp), objects_large);

  const double yy_small = objects_small.objects.front().pose_covariance[XYZRPY_COV_IDX::Y_Y];
  const double yy_large = objects_large.objects.front().pose_covariance[XYZRPY_COV_IDX::Y_Y];
  EXPECT_GT(yy_large, yy_small);
  EXPECT_NEAR(yy_large - yy_small, (yaw_var_large - yaw_var_small) * range * range, 1e-9);
}

// The ego position stddevs add isotropically (independent of range), covering the pos_stddev_x/y
// knobs of the localization error model.
TEST(AddOdometryUncertainty, PositionErrorAddsIsotropically)
{
  const rclcpp::Time stamp(100, 0, RCL_ROS_TIME);
  const double pos_var = 0.3 * 0.3;

  auto objects = makeSingleObject(10.0, 0.0, stamp);
  addOdometryUncertainty(makeEgoOdometry(pos_var, pos_var, 0.0, stamp), objects);

  const auto & cov = objects.objects.front().pose_covariance;
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::X_X], kInitPosVar + pos_var, 1e-9);
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::Y_Y], kInitPosVar + pos_var, 1e-9);
  EXPECT_NEAR(cov[XYZRPY_COV_IDX::YAW_YAW], kInitYawVar, 1e-12);  // no yaw error => yaw untouched
}
