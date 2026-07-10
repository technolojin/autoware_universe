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

// Tests for the ego-localization-driven position process noise injected during prediction
// (MotionModel::addOdometryProcessNoise, exercised through CVMotionModel). The model is
//   dQ_pos(dt, r) = (dt / tau) * [ pos_cov + yaw_var * r^2 * t_hat t_hat^T ]   (ego POSE error)
//                 +  dt^2      *   vel_cov                                     (ego MOTION)
// so the growth must: (a) be a function of dt that vanishes at dt = 0; (b) project an ego yaw
// error to a lateral (tangential) object error growing with range (and vanishing as the ego
// reaches the object). A single prediction step (dt <= dt_max = 0.11 s) is used so the per-step
// contribution is exact and easy to check.

#include "autoware/multi_object_tracker/tracker/motion_model/cv_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/ego_uncertainty.hpp"

#include <Eigen/Core>
#include <autoware_utils_geometry/msg/covariance.hpp>
#include <rclcpp/time.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <gtest/gtest.h>

#include <array>

namespace
{
using autoware::multi_object_tracker::CVMotionModel;
using autoware::multi_object_tracker::EgoUncertainty;
using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;

const rclcpp::Time kT0(100, 0, RCL_ROS_TIME);
constexpr double kInitPosVar = 1e-4;

// A stationary CV object at (x, y) with zero own process noise, so the only covariance growth
// during prediction comes from the ego-localization term under test.
CVMotionModel makeStationaryObject(double x, double y)
{
  CVMotionModel model;
  model.setMotionParams(0.0, 0.0, 0.0, 0.0);  // no own process noise
  std::array<double, 36> pose_cov{};
  pose_cov[XYZRPY_COV_IDX::X_X] = kInitPosVar;
  pose_cov[XYZRPY_COV_IDX::Y_Y] = kInitPosVar;
  std::array<double, 36> twist_cov{};
  model.initialize(kT0, x, y, pose_cov, 0.0, 0.0, twist_cov);  // zero velocity
  return model;
}

// Ego at (ego_x, ego_y) with a pure heading (yaw) uncertainty, leaked over correlation time tau.
EgoUncertainty makeYawEgo(double ego_x, double ego_y, double yaw_stddev, double tau)
{
  EgoUncertainty ego;
  ego.ego_x = ego_x;
  ego.ego_y = ego_y;
  ego.yaw_var = yaw_stddev * yaw_stddev;
  ego.inv_correlation_time = 1.0 / tau;
  return ego;
}

// Predicted position covariance element at kT0 + dt.
double predictedCov(const CVMotionModel & model, double dt, int cov_idx)
{
  geometry_msgs::msg::Pose pose;
  geometry_msgs::msg::Twist twist;
  std::array<double, 36> pose_cov{};
  std::array<double, 36> twist_cov{};
  model.getPredictedState(
    kT0 + rclcpp::Duration::from_seconds(dt), pose, pose_cov, twist, twist_cov);
  return pose_cov[cov_idx];
}
}  // namespace

// An ego heading error projects to a lateral (tangential) object-position growth of
// (dt/tau)*yaw_var*r^2 for an object straight ahead; the longitudinal (radial) axis is untouched.
// Calibration check: yaw_stddev = 0.8 deg at r = 90 m gives ~1.3 m lateral stddev.
TEST(EgoProcessNoise, YawErrorProjectsToLateralGrowthByRange)
{
  const double range = 90.0;
  const double yaw_stddev = 0.8 * M_PI / 180.0;  // 0.8 deg
  const double tau = 1.0;
  const double dt = 0.1;  // single prediction step

  const auto model = makeStationaryObject(range, 0.0);  // straight ahead of the ego at origin
  auto model_ego = makeStationaryObject(range, 0.0);
  model_ego.setEgoUncertainty(makeYawEgo(0.0, 0.0, yaw_stddev, tau));

  const double yy_base = predictedCov(model, dt, XYZRPY_COV_IDX::Y_Y);
  const double yy_ego = predictedCov(model_ego, dt, XYZRPY_COV_IDX::Y_Y);
  const double xx_ego = predictedCov(model_ego, dt, XYZRPY_COV_IDX::X_X);

  const double expected = (dt / tau) * yaw_stddev * yaw_stddev * range * range;
  EXPECT_NEAR(
    yy_ego - yy_base, expected, expected * 1e-3);  // lateral grows by (dt/tau)*yaw_var*r^2
  EXPECT_NEAR(xx_ego, kInitPosVar, 1e-6);          // radial (longitudinal) untouched
  // Sanity: the lateral stddev accrued matches the ~1.3 m headline figure.
  EXPECT_NEAR(std::sqrt(yaw_stddev * yaw_stddev * range * range), 1.2566, 1e-3);
}

// The lateral growth scales with r^2, so it shrinks toward zero as the ego approaches the object.
TEST(EgoProcessNoise, LateralGrowthShrinksAsEgoApproaches)
{
  const double yaw_stddev = 0.8 * M_PI / 180.0;
  const double tau = 1.0;
  const double dt = 0.1;

  auto far = makeStationaryObject(90.0, 0.0);
  auto near = makeStationaryObject(10.0, 0.0);
  far.setEgoUncertainty(makeYawEgo(0.0, 0.0, yaw_stddev, tau));
  near.setEgoUncertainty(makeYawEgo(0.0, 0.0, yaw_stddev, tau));

  const double growth_far = predictedCov(far, dt, XYZRPY_COV_IDX::Y_Y) - kInitPosVar;
  const double growth_near = predictedCov(near, dt, XYZRPY_COV_IDX::Y_Y) - kInitPosVar;

  EXPECT_GT(growth_far, growth_near);
  EXPECT_NEAR(growth_far / growth_near, (90.0 * 90.0) / (10.0 * 10.0), 1e-3);  // ratio of r^2 = 81
}

// The process noise is a function of dt and vanishes at dt = 0 (the pose term is linear in dt for a
// single step): halving dt halves the growth, and dt -> 0 adds nothing.
TEST(EgoProcessNoise, GrowthIsProportionalToDtAndZeroAtZero)
{
  const double yaw_stddev = 0.8 * M_PI / 180.0;
  const double tau = 1.0;

  auto model = makeStationaryObject(90.0, 0.0);
  model.setEgoUncertainty(makeYawEgo(0.0, 0.0, yaw_stddev, tau));

  const double growth_full = predictedCov(model, 0.10, XYZRPY_COV_IDX::Y_Y) - kInitPosVar;
  const double growth_half = predictedCov(model, 0.05, XYZRPY_COV_IDX::Y_Y) - kInitPosVar;
  const double growth_zero = predictedCov(model, 1e-9, XYZRPY_COV_IDX::Y_Y) - kInitPosVar;

  EXPECT_NEAR(growth_half / growth_full, 0.5, 1e-3);  // linear in dt
  EXPECT_NEAR(growth_zero, 0.0, 1e-9);                // zero dt => zero process noise
}

// The yaw-lever growth is directed tangentially: for an object off to the ego's side the growth
// lands on the longitudinal (x) axis instead of the lateral (y) one.
TEST(EgoProcessNoise, GrowthDirectionFollowsLineOfSight)
{
  const double range = 50.0;
  const double yaw_stddev = 0.8 * M_PI / 180.0;
  const double tau = 1.0;
  const double dt = 0.1;

  auto model = makeStationaryObject(0.0, range);  // directly to the ego's left
  model.setEgoUncertainty(makeYawEgo(0.0, 0.0, yaw_stddev, tau));

  const double expected = (dt / tau) * yaw_stddev * yaw_stddev * range * range;
  EXPECT_NEAR(
    predictedCov(model, dt, XYZRPY_COV_IDX::X_X) - kInitPosVar, expected, expected * 1e-3);
  EXPECT_NEAR(predictedCov(model, dt, XYZRPY_COV_IDX::Y_Y), kInitPosVar, 1e-6);
}

// A zero (default) ego bundle leaves prediction unchanged (backward compatible / opt-in).
TEST(EgoProcessNoise, ZeroEgoUncertaintyIsNoOp)
{
  const double dt = 0.1;
  auto model = makeStationaryObject(90.0, 0.0);  // no setEgoUncertainty() call
  EXPECT_NEAR(predictedCov(model, dt, XYZRPY_COV_IDX::Y_Y), kInitPosVar, 1e-9);
  EXPECT_NEAR(predictedCov(model, dt, XYZRPY_COV_IDX::X_X), kInitPosVar, 1e-9);
}
