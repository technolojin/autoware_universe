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

#include "autoware/multi_object_tracker/object_model/object_model.hpp"
#include "autoware/multi_object_tracker/object_model/shapes.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/bicycle_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/shape_model/vehicle_shape_model.hpp"
#include "autoware/multi_object_tracker/tracker/update/vehicle_update_strategy.hpp"

#include <autoware_utils_geometry/msg/covariance.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
namespace
{
geometry_msgs::msg::Pose makePose(double x, double y, double yaw)
{
  geometry_msgs::msg::Pose p;
  p.position.x = x;
  p.position.y = y;
  p.position.z = 0.0;
  p.orientation.x = 0.0;
  p.orientation.y = 0.0;
  p.orientation.z = std::sin(yaw / 2.0);
  p.orientation.w = std::cos(yaw / 2.0);
  return p;
}

types::DynamicObject makeBox(double cx, double cy, double yaw, double len, double wid)
{
  types::DynamicObject o;
  o.pose = makePose(cx, cy, yaw);
  o.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  o.shape.dimensions.x = len;
  o.shape.dimensions.y = wid;
  o.shape.dimensions.z = 1.5;
  o.area = len * wid;
  return o;
}

// Build a POLYGON cluster with an identity pose, so footprint coords are already map-frame.
types::DynamicObject makePolyCluster(const std::vector<std::pair<double, double>> & pts)
{
  types::DynamicObject o;
  o.pose = makePose(0.0, 0.0, 0.0);
  o.shape.type = autoware_perception_msgs::msg::Shape::POLYGON;
  o.shape.dimensions.z = 1.5;
  double a = 0.0;
  const size_t n = pts.size();
  for (size_t i = 0; i < n; ++i) {
    geometry_msgs::msg::Point32 p;
    p.x = static_cast<float>(pts[i].first);
    p.y = static_cast<float>(pts[i].second);
    p.z = 0.0f;
    o.shape.footprint.points.push_back(p);
    const auto & q0 = pts[i];
    const auto & q1 = pts[(i + 1) % n];
    a += q0.first * q1.second - q1.first * q0.second;
  }
  o.area = std::abs(0.5 * a);
  return o;
}

geometry_msgs::msg::Point makeEgo(double x, double y)
{
  geometry_msgs::msg::Point e;
  e.x = x;
  e.y = y;
  e.z = 0.0;
  return e;
}
}  // namespace

// --- analyzePolygonMeasurement (observation-only corner) ---------------------------------------

namespace
{
double nearestVertexDist(const types::DynamicObject & cluster, double tx, double ty)
{
  double best = std::numeric_limits<double>::max();
  for (const auto & p : cluster.shape.footprint.points) {
    best = std::min(best, std::hypot(static_cast<double>(p.x) - tx, static_cast<double>(p.y) - ty));
  }
  return best;
}
}  // namespace

// A clean L: the visible corner is the cluster's ego-facing extreme read in the predicted box
// frame. Ego is behind-and-right of the box, so the rear-right corner (8, -1) is measured, and the
// association reports rear (is_front=false) / right (s_lat=-1). Both faces are resolved -> the
// covariance is a valid, well-conditioned 2x2.
TEST(AnalyzePolygonMeasurement, LShapeCornerIsBoxFrameExtreme)
{
  const auto cluster = makePolyCluster({{8, -1}, {12, -1}, {12, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto m = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, -10.0), pred);

  EXPECT_TRUE(m.has_corner);
  EXPECT_NEAR(m.corner.x, 8.0, 1e-6);
  EXPECT_NEAR(m.corner.y, -1.0, 1e-6);
  EXPECT_FALSE(m.is_front);
  EXPECT_DOUBLE_EQ(m.s_lat, -1.0);
  // Covariance is a valid 2x2 (PD): positive diagonal and positive determinant.
  EXPECT_GT(m.corner_cov[0], 0.0);
  EXPECT_GT(m.corner_cov[3], 0.0);
  EXPECT_GT(m.corner_cov[0] * m.corner_cov[3] - m.corner_cov[1] * m.corner_cov[2], 0.0);
}

// Rounding-immunity: the true corner (8, -1) is cut away (no vertex sits there). Because the corner
// is read as the box-frame EXTENT intersection — min-x of the left face, min-y of the bottom face —
// it still lands exactly on the true corner, whereas the closest SAMPLED vertex (what a nearest-
// vertex rule would anchor on) is far off. This is the core of the prior-driven extraction.
TEST(AnalyzePolygonMeasurement, RoundedCornerRecoversBoxExtreme)
{
  // Long clean left face (x=8) and bottom face (y=-1); the (8,-1) corner is replaced by a chamfer.
  const auto cluster = makePolyCluster({
    {8.7, -1},
    {9, -1},
    {10, -1},
    {11, -1},
    {12, -1},  // bottom face (y = -1)
    {12, 1},   // right (occluded)
    {8, 1},
    {8, 0.5},
    {8, 0.0},
    {8, -0.4},     // left face (x = 8)
    {8.3, -0.75},  // chamfer (rounding) — no vertex at (8,-1)
  });
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto m = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, -10.0), pred);

  ASSERT_TRUE(m.has_corner);
  const double recovered_err = std::hypot(m.corner.x - 8.0, m.corner.y + 1.0);
  EXPECT_LT(recovered_err, 1e-6);
  EXPECT_GT(nearestVertexDist(cluster, 8.0, -1.0), 0.35);
  EXPECT_LT(recovered_err, nearestVertexDist(cluster, 8.0, -1.0));
}

// A single visible face (thin cluster, only the rear face resolved) still yields a corner, but with
// an anisotropic covariance: the axis the face localizes (here longitudinal / x) is tight, while
// the unobserved axis (lateral / y — we cannot see how far the body extends sideways) is left
// nearly unconstrained. The EKF then corrects only the well-observed direction.
TEST(AnalyzePolygonMeasurement, SingleFaceGivesAnisotropicCovariance)
{
  const auto cluster = makePolyCluster({{8, -1}, {8.2, -1}, {8.2, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto m = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, 0.0), pred);

  ASSERT_TRUE(m.has_corner);
  EXPECT_NEAR(m.corner.x, 8.0, 1e-6);  // longitudinal face position is observed
  // yy (lateral, unobserved) is orders of magnitude larger than xx (longitudinal, observed).
  EXPECT_GT(m.corner_cov[3], 100.0 * m.corner_cov[0]);
  // Still a valid PD covariance.
  EXPECT_GT(m.corner_cov[0] * m.corner_cov[3] - m.corner_cov[1] * m.corner_cov[2], 0.0);
}

// The ego-facing side selects which box corner is measured: moving ego around the same box yields
// each of the four quadrants, with the corner at the matching extent and the matching association.
TEST(AnalyzePolygonMeasurement, FacingSelectsEgoSideCorner)
{
  const auto cluster = makePolyCluster({{8, -1}, {12, -1}, {12, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);

  const auto rear_right = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, -10.0), pred);
  EXPECT_NEAR(rear_right.corner.x, 8.0, 1e-6);
  EXPECT_NEAR(rear_right.corner.y, -1.0, 1e-6);
  EXPECT_FALSE(rear_right.is_front);
  EXPECT_DOUBLE_EQ(rear_right.s_lat, -1.0);

  const auto front_right = shapes::analyzePolygonMeasurement(cluster, makeEgo(20.0, -10.0), pred);
  EXPECT_NEAR(front_right.corner.x, 12.0, 1e-6);
  EXPECT_NEAR(front_right.corner.y, -1.0, 1e-6);
  EXPECT_TRUE(front_right.is_front);
  EXPECT_DOUBLE_EQ(front_right.s_lat, -1.0);

  const auto front_left = shapes::analyzePolygonMeasurement(cluster, makeEgo(20.0, 10.0), pred);
  EXPECT_NEAR(front_left.corner.x, 12.0, 1e-6);
  EXPECT_NEAR(front_left.corner.y, 1.0, 1e-6);
  EXPECT_TRUE(front_left.is_front);
  EXPECT_DOUBLE_EQ(front_left.s_lat, 1.0);

  const auto rear_left = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, 10.0), pred);
  EXPECT_NEAR(rear_left.corner.x, 8.0, 1e-6);
  EXPECT_NEAR(rear_left.corner.y, 1.0, 1e-6);
  EXPECT_FALSE(rear_left.is_front);
  EXPECT_DOUBLE_EQ(rear_left.s_lat, 1.0);
}

// A rounded body (12-gon) is no longer rejected as "not a corner": the prior asserts a box, so we
// report its ego-facing bounding corner. The EKF innovation gate, not a shape classifier, is what
// guards against a genuinely bad cluster.
TEST(AnalyzePolygonMeasurement, RoundedBodyStillYieldsBoundingCorner)
{
  std::vector<std::pair<double, double>> pts;
  for (int i = 0; i < 12; ++i) {
    const double th = 2.0 * M_PI * static_cast<double>(i) / 12.0;
    pts.push_back({10.0 + 1.5 * std::cos(th), 1.5 * std::sin(th)});
  }
  const auto cluster = makePolyCluster(pts);
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto m = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, -10.0), pred);

  ASSERT_TRUE(m.has_corner);
  EXPECT_NEAR(m.corner.x, 8.5, 1e-6);   // min-x extent of the blob
  EXPECT_NEAR(m.corner.y, -1.5, 1e-6);  // min-y extent of the blob
}

// --- wheel-anchor update from a reconstructed corner ---------------------------------------------

// The corner measurement drives the EKF through the wheel-anchor update: VehicleTracker
// reconstructs the rear/front FACE center from the observed corner (subtracting the prior
// half-width offset s_lat * (width/2) * n at the predicted yaw), then calls
// updateStatePoseRear/Front. An observed rear corner offset laterally from the prediction pulls the
// body toward it (between prediction and observation, never past it). This reproduces that
// tracker-side reconstruction to prove the wheel-anchor update moves the filter toward the
// observation without the prior being pre-fused into the measured value.
TEST(UpdateStatePoseRear, MovesEstimateTowardReconstructedCorner)
{
  BicycleMotionModel model;
  const rclcpp::Time t(0, 0, RCL_ROS_TIME);
  std::array<double, 36> pose_cov{};
  using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  pose_cov[XYZRPY_COV_IDX::X_X] = 1.0;
  pose_cov[XYZRPY_COV_IDX::Y_Y] = 1.0;
  // Box centered at (10,0), yaw 0, length 4. Rear face center is at x=8; rear-left corner (width 2)
  // is (8, 1).
  ASSERT_TRUE(model.initialize(t, 10.0, 0.0, 0.0, pose_cov, 0.0, 1.0, 0.0, 1.0, 4.0));

  geometry_msgs::msg::Pose pose0;
  std::array<double, 36> c0{};
  geometry_msgs::msg::Twist tw0;
  ASSERT_TRUE(model.getPredictedState(t, pose0, c0, tw0, c0));
  EXPECT_NEAR(pose0.position.y, 0.0, 1e-6);

  // Observe the rear-left corner shifted +0.4 in y -> (8, 1.4). Reconstruct the rear face center as
  // the tracker does: face = corner - s_lat * (width/2) * n, n = (-sin yaw, cos yaw); at yaw 0 this
  // is (8, 0.4). Tight covariance (packed into the x/y block) so the gain is sizable.
  const double yaw = model.getYawState();
  constexpr double s_lat = 1.0;
  constexpr double half_w = 0.5 * 2.0;
  const double face_x = 8.0 - s_lat * half_w * (-std::sin(yaw));
  const double face_y = 1.4 - s_lat * half_w * (std::cos(yaw));
  std::array<double, 36> corner_cov{};
  corner_cov[XYZRPY_COV_IDX::X_X] = 0.01;
  corner_cov[XYZRPY_COV_IDX::Y_Y] = 0.01;
  ASSERT_TRUE(model.updateStatePoseRear(face_x, face_y, corner_cov));

  geometry_msgs::msg::Pose pose1;
  std::array<double, 36> c1{};
  geometry_msgs::msg::Twist tw1;
  ASSERT_TRUE(model.getPredictedState(t, pose1, c1, tw1, c1));
  EXPECT_GT(pose1.position.y, pose0.position.y);  // moved toward the observation
  EXPECT_LT(pose1.position.y, 0.4);               // but not past it
  EXPECT_NEAR(pose1.position.x, 10.0, 0.1);       // longitudinal ~unchanged
}

TEST(UpdateStatePoseRear, MahalanobisGateRejectsGrossOutlier)
{
  BicycleMotionModel model;
  const rclcpp::Time t(0, 0, RCL_ROS_TIME);
  std::array<double, 36> pose_cov{};
  using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  pose_cov[XYZRPY_COV_IDX::X_X] = 1.0;
  pose_cov[XYZRPY_COV_IDX::Y_Y] = 1.0;
  // Box centered at (10,0), yaw 0, length 4. Rear face center is at (8,0).
  ASSERT_TRUE(model.initialize(t, 10.0, 0.0, 0.0, pose_cov, 0.0, 1.0, 0.0, 1.0, 4.0));

  geometry_msgs::msg::Pose pose0;
  std::array<double, 36> c0{};
  geometry_msgs::msg::Twist tw0;
  ASSERT_TRUE(model.getPredictedState(t, pose0, c0, tw0, c0));

  // Gross mis-association: a reconstructed rear face center 20 m off laterally with a tight
  // covariance (the regime that previously slammed the estimate into a wrong / merged cluster). The
  // Mahalanobis gate must REJECT it (return false) and leave the state untouched. This locks in the
  // outlier protection that was lost when updateStatePoseCorner was folded into the front/rear path.
  std::array<double, 36> corner_cov{};
  corner_cov[XYZRPY_COV_IDX::X_X] = 0.01;
  corner_cov[XYZRPY_COV_IDX::Y_Y] = 0.01;
  EXPECT_FALSE(model.updateStatePoseRear(8.0, 20.0, corner_cov));

  geometry_msgs::msg::Pose pose1;
  std::array<double, 36> c1{};
  geometry_msgs::msg::Twist tw1;
  ASSERT_TRUE(model.getPredictedState(t, pose1, c1, tw1, c1));
  EXPECT_NEAR(pose1.position.x, pose0.position.x, 1e-9);  // state unchanged: outlier rejected
  EXPECT_NEAR(pose1.position.y, pose0.position.y, 1e-9);
}

}  // namespace autoware::multi_object_tracker
