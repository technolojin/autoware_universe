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
geometry_msgs::msg::Point makePoint(double x, double y)
{
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = 0.0;
  return p;
}

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

double wrapToPi(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
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

// A clean L: the corner is the intersection of the two visible faces; here that coincides with the
// sampled vertex (8, -1). Heading runs along the long (bottom) face.
TEST(AnalyzePolygonMeasurement, LShapeCornerIsLineIntersection)
{
  const auto cluster = makePolyCluster({{8, -1}, {12, -1}, {12, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto m = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, -10.0), pred);

  EXPECT_TRUE(m.has_corner);
  EXPECT_NEAR(m.corner.x, 8.0, 1e-6);
  EXPECT_NEAR(m.corner.y, -1.0, 1e-6);
  EXPECT_TRUE(m.has_yaw);
  EXPECT_NEAR(wrapToPi(m.yaw), 0.0, 0.05);
  // Covariance is a valid 2x2 (PD): positive diagonal and positive determinant.
  EXPECT_GT(m.corner_cov[0], 0.0);
  EXPECT_GT(m.corner_cov[3], 0.0);
  EXPECT_GT(m.corner_cov[0] * m.corner_cov[3] - m.corner_cov[1] * m.corner_cov[2], 0.0);
}

// Rounding-immunity: the true corner (8, -1) is cut away (no vertex sits there). The intersection
// of the two fitted faces still lands on the true corner, whereas the closest SAMPLED vertex — what
// the old nearest-point rule would have anchored on — is far off. This is the core fix.
TEST(AnalyzePolygonMeasurement, RoundedCornerRecoversTrueCorner)
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
  // The recovered corner sits on the true corner well within the chamfer cut-out; the closest
  // SAMPLED vertex (what the old nearest-point rule anchored on) is far further away. The residual
  // bias here is the single synthetic chamfer point pulling one fit — real multi-point rounding
  // averages out better. This contrast is the core fix: face intersection, not a vertex apex.
  const double recovered_err = std::hypot(m.corner.x - 8.0, m.corner.y + 1.0);
  EXPECT_LT(recovered_err, 0.25);
  EXPECT_GT(nearestVertexDist(cluster, 8.0, -1.0), 0.35);
  EXPECT_LT(recovered_err, nearestVertexDist(cluster, 8.0, -1.0));
}

// A single visible face (thin rear cluster, one face only) resolves no corner: has_corner=false.
// The caller keeps the (separate) single-face / weak path; we do not fabricate a corner.
TEST(AnalyzePolygonMeasurement, SingleFaceHasNoCorner)
{
  const auto cluster = makePolyCluster({{8, -1}, {8.2, -1}, {8.2, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto m = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, 0.0), pred);

  EXPECT_FALSE(m.has_corner);
}

// A rounded body (12-gon, 30 deg turns) has no two statistically distinct faces: no corner.
TEST(AnalyzePolygonMeasurement, RoundedBodyHasNoCorner)
{
  std::vector<std::pair<double, double>> pts;
  for (int i = 0; i < 12; ++i) {
    const double th = 2.0 * M_PI * static_cast<double>(i) / 12.0;
    pts.push_back({10.0 + 1.5 * std::cos(th), 1.5 * std::sin(th)});
  }
  const auto cluster = makePolyCluster(pts);
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto m = shapes::analyzePolygonMeasurement(cluster, makeEgo(0.0, -10.0), pred);

  EXPECT_FALSE(m.has_corner);
}

// --- updateStatePoseCorner -----------------------------------------------------------------------

// The corner measurement drives the EKF: an observed rear corner offset laterally from the
// predicted rear corner pulls the body toward it (between prediction and observation, never past
// it), with the prior width living only in the predicted measurement. Proves the observation-only
// mean moves the filter without being pre-fused with the prior.
TEST(UpdateStatePoseCorner, MovesEstimateTowardObservedCorner)
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

  // Observe the rear-left corner shifted +0.4 in y (small, tight covariance so the gain is
  // sizable).
  const std::array<double, 4> corner_cov{0.01, 0.0, 0.0, 0.01};
  ASSERT_TRUE(model.updateStatePoseCorner(8.0, 1.4, corner_cov, /*is_front=*/false, 1.0, 2.0));

  geometry_msgs::msg::Pose pose1;
  std::array<double, 36> c1{};
  geometry_msgs::msg::Twist tw1;
  ASSERT_TRUE(model.getPredictedState(t, pose1, c1, tw1, c1));
  EXPECT_GT(pose1.position.y, pose0.position.y);  // moved toward the observation
  EXPECT_LT(pose1.position.y, 0.4);               // but not past it
  EXPECT_NEAR(pose1.position.x, 10.0, 0.1);       // longitudinal ~unchanged
}

// A grossly inconsistent corner (many sigma off) is rejected by the Mahalanobis gate: the update
// returns false and the state is left untouched, so a mis-association cannot corrupt the track.
TEST(UpdateStatePoseCorner, GatesOutGrossOutlier)
{
  BicycleMotionModel model;
  const rclcpp::Time t(0, 0, RCL_ROS_TIME);
  std::array<double, 36> pose_cov{};
  using autoware_utils_geometry::xyzrpy_covariance_index::XYZRPY_COV_IDX;
  pose_cov[XYZRPY_COV_IDX::X_X] = 1.0;
  pose_cov[XYZRPY_COV_IDX::Y_Y] = 1.0;
  ASSERT_TRUE(model.initialize(t, 10.0, 0.0, 0.0, pose_cov, 0.0, 1.0, 0.0, 1.0, 4.0));

  geometry_msgs::msg::Pose pose0;
  std::array<double, 36> c0{};
  geometry_msgs::msg::Twist tw0;
  ASSERT_TRUE(model.getPredictedState(t, pose0, c0, tw0, c0));

  // Predicted rear-left corner is (8, 1); observe it 30 m away with a tight covariance -> the
  // innovation is enormous in sigma and must be gated out.
  const std::array<double, 4> corner_cov{0.01, 0.0, 0.0, 0.01};
  EXPECT_FALSE(model.updateStatePoseCorner(8.0, 31.0, corner_cov, /*is_front=*/false, 1.0, 2.0));

  geometry_msgs::msg::Pose pose1;
  std::array<double, 36> c1{};
  geometry_msgs::msg::Twist tw1;
  ASSERT_TRUE(model.getPredictedState(t, pose1, c1, tw1, c1));
  EXPECT_NEAR(pose1.position.x, pose0.position.x, 1e-9);  // state untouched
  EXPECT_NEAR(pose1.position.y, pose0.position.y, 1e-9);
}

// --- associateCornerToPrediction ---------------------------------------------------------------

// Each observed corner is associated to the predicted box quadrant it falls in: front/rear from the
// longitudinal sign, left/right (s_lat) from the lateral sign. Box at (10,0), yaw 0, 4 x 2.
TEST(AssociateCornerToPrediction, ResolvesQuadrant)
{
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);

  const auto front_left = associateCornerToPrediction(makePoint(12.0, 1.0), pred);
  EXPECT_TRUE(front_left.is_front);
  EXPECT_DOUBLE_EQ(front_left.s_lat, 1.0);

  const auto front_right = associateCornerToPrediction(makePoint(12.0, -1.0), pred);
  EXPECT_TRUE(front_right.is_front);
  EXPECT_DOUBLE_EQ(front_right.s_lat, -1.0);

  const auto rear_left = associateCornerToPrediction(makePoint(8.0, 1.0), pred);
  EXPECT_FALSE(rear_left.is_front);
  EXPECT_DOUBLE_EQ(rear_left.s_lat, 1.0);

  const auto rear_right = associateCornerToPrediction(makePoint(8.0, -1.0), pred);
  EXPECT_FALSE(rear_right.is_front);
  EXPECT_DOUBLE_EQ(rear_right.s_lat, -1.0);
}

// Association respects the predicted yaw: with the body rotated 90 deg, the longitudinal axis is
// +y, so a corner displaced in +y is "front" and one in +x (the body's right) is rear-right.
TEST(AssociateCornerToPrediction, UsesPredictedYaw)
{
  const auto pred = makeBox(0.0, 0.0, M_PI_2, 4.0, 2.0);

  const auto a = associateCornerToPrediction(makePoint(0.5, 3.0), pred);
  EXPECT_TRUE(a.is_front);  // +y is forward when yaw = 90 deg

  // Body lateral axis n = (-sin, cos) = (-1, 0); a +x displacement projects onto n as negative.
  const auto b = associateCornerToPrediction(makePoint(0.5, -3.0), pred);
  EXPECT_FALSE(b.is_front);
  EXPECT_DOUBLE_EQ(b.s_lat, -1.0);
}

// --- one-sided (grow-only) dimension filter ----------------------------------------------------

// growWidth is a lower-bound filter: a wider observation grows the tracked width (capped at the
// size limit), a narrower one never shrinks it, and a non-positive observation is a no-op.
TEST(VehicleShapeModelGrowWidth, GrowsOnlyAndClamps)
{
  VehicleShapeModel shape_model(object_model::normal_vehicle);
  shape_model.init(makeBox(0.0, 0.0, 0.0, 4.0, 2.0));  // width seeded at 2.0
  ASSERT_DOUBLE_EQ(shape_model.getWidth(), 2.0);

  shape_model.growWidth(3.0);  // wider -> grows
  EXPECT_DOUBLE_EQ(shape_model.getWidth(), 3.0);

  shape_model.growWidth(1.5);  // narrower -> held
  EXPECT_DOUBLE_EQ(shape_model.getWidth(), 3.0);

  shape_model.growWidth(100.0);  // beyond the limit -> capped at width_max (5.0 for NormalVehicle)
  EXPECT_DOUBLE_EQ(shape_model.getWidth(), 5.0);

  shape_model.growWidth(0.0);  // non-positive -> no-op
  EXPECT_DOUBLE_EQ(shape_model.getWidth(), 5.0);
}

}  // namespace autoware::multi_object_tracker
