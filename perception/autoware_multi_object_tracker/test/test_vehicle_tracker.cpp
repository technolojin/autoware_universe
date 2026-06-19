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

#include "autoware/multi_object_tracker/object_model/shapes.hpp"
#include "autoware/multi_object_tracker/tracker/motion_model/bicycle_motion_model.hpp"
#include "autoware/multi_object_tracker/tracker/update/vehicle_update_strategy.hpp"

#include <autoware_utils_geometry/msg/covariance.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
namespace
{
// Matches the values wired into VehicleTracker::updateWheelKinematics.
constexpr double kAlpha = 0.2;  // balance_alpha
constexpr double kBeta = 0.3;   // corner_residual_beta

geometry_msgs::msg::Point makePoint(double x, double y)
{
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = 0.0;
  return p;
}

// With yaw = 0 the body lateral axis is +y, so the lateral offset d equals (anchor.y - center.y)
// and the corrected anchor keeps x and moves only y. tracker_center is the origin throughout.
WheelAnchorLateral run(double tracker_width, double polygon_width, double anchor_y)
{
  return correctWheelAnchorLateral(
    0.0, tracker_width, makePoint(0.0, 0.0), polygon_width, makePoint(5.0, anchor_y), kAlpha,
    kBeta);
}
}  // namespace

// Equal widths: no slack, no gap -> anchor untouched, no added variance.
TEST(CorrectWheelAnchorLateral, EqualWidthIsNoOp)
{
  const auto r = run(2.0, 2.0, 0.5);
  EXPECT_DOUBLE_EQ(r.anchor.x, 5.0);
  EXPECT_DOUBLE_EQ(r.anchor.y, 0.5);
  EXPECT_DOUBLE_EQ(r.var_lat, 0.0);
}

// Polygon narrower than the tracker (partial view): anchor kept, worst-case lateral offset added
// as variance, var = ((w_t - w_p) / 2)^2.
TEST(CorrectWheelAnchorLateral, NarrowPolygonAddsVarianceOnly)
{
  const auto r = run(2.0, 1.0, 0.5);
  EXPECT_DOUBLE_EQ(r.anchor.x, 5.0);
  EXPECT_DOUBLE_EQ(r.anchor.y, 0.5);  // anchor unchanged
  EXPECT_DOUBLE_EQ(r.var_lat, 0.25);  // (0.5)^2
}

// Wide polygon, observed center on the body axis: nothing to pull, full slack uncertainty.
TEST(CorrectWheelAnchorLateral, WideCenteredHoldsAnchorMaxVariance)
{
  const double slack = 1.0;  // (4 - 2) / 2
  const auto r = run(2.0, 4.0, 0.0);
  EXPECT_DOUBLE_EQ(r.anchor.y, 0.0);
  EXPECT_DOUBLE_EQ(r.var_lat, slack * slack);  // std = slack
}

// Wide polygon, observed center inside the slack: anchor held near the tracker (slope alpha), so it
// is pulled back toward the body axis rather than snapped to the observed center.
TEST(CorrectWheelAnchorLateral, WideContainedPullsTowardTracker)
{
  const double slack = 1.0;
  const double d = 0.5;  // < slack -> contained
  const auto r = run(2.0, 4.0, d);
  EXPECT_DOUBLE_EQ(r.anchor.y, kAlpha * d);  // 0.1, pulled in from 0.5
  const double t = d / slack;
  const double std_lat = slack * (1.0 - (1.0 - kBeta) * t);
  EXPECT_DOUBLE_EQ(r.var_lat, std_lat * std_lat);
}

// Wide polygon, observed center beyond the slack: a corner is exposed, anchor follows it (unit
// slope) and the added variance shrinks toward beta * slack.
TEST(CorrectWheelAnchorLateral, WideUncontainedFollowsCorner)
{
  const double slack = 1.0;
  const double d = 2.0;  // > slack -> corner exposed
  const auto r = run(2.0, 4.0, d);
  EXPECT_DOUBLE_EQ(r.anchor.y, (d - slack) + kAlpha * slack);      // 1.2
  EXPECT_DOUBLE_EQ(r.var_lat, (kBeta * slack) * (kBeta * slack));  // std = beta * slack
}

// The dead-zone is continuous at |d| = slack: both branches agree on anchor and variance.
TEST(CorrectWheelAnchorLateral, ContinuousAtBoundary)
{
  const double slack = 1.0;
  const double eps = 1e-6;
  const auto inside = run(2.0, 4.0, slack - eps);
  const auto outside = run(2.0, 4.0, slack + eps);
  EXPECT_NEAR(inside.anchor.y, outside.anchor.y, 1e-4);
  EXPECT_NEAR(inside.var_lat, outside.var_lat, 1e-4);
}

// Sign symmetry: a negative lateral offset mirrors the positive case.
TEST(CorrectWheelAnchorLateral, SignSymmetry)
{
  const auto pos = run(2.0, 4.0, 1.5);
  const auto neg = run(2.0, 4.0, -1.5);
  EXPECT_DOUBLE_EQ(pos.anchor.y, -neg.anchor.y);
  EXPECT_DOUBLE_EQ(pos.var_lat, neg.var_lat);
}

// --- analyzePolygonGeometry --------------------------------------------------------------------

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

double wrapToPi(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
}
}  // namespace

// A clean L-shape viewed from a corner: recover the near corner, the long-edge yaw, the width, and
// flag no inflation.
TEST(AnalyzePolygonGeometry, LShapeRecoversCorner)
{
  const auto cluster = makePolyCluster({{8, -1}, {12, -1}, {12, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(0.0, -10.0), pred);

  EXPECT_TRUE(g.has_corner);
  EXPECT_NEAR(g.near_corner.x, 8.0, 1e-6);
  EXPECT_NEAR(g.near_corner.y, -1.0, 1e-6);
  EXPECT_TRUE(g.yaw_cue_valid);
  EXPECT_NEAR(wrapToPi(g.long_edge_dir), 0.0, 0.1);  // long side runs along x
  EXPECT_NEAR(g.observed_width, 2.0, 1e-6);
  EXPECT_EQ(g.inflation, shapes::PolygonInflation::NONE);
  EXPECT_GT(g.trust, 0.5);
}

// The strategy reconstructs the observed REAR face center from the near corner + tracked width.
TEST(AnalyzePolygonGeometry, StrategyReconstructsRearFaceCenter)
{
  const auto cluster = makePolyCluster({{8, -1}, {12, -1}, {12, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(0.0, -10.0), pred);
  const auto s = determineUpdateStrategy(g, pred, 2.0);

  EXPECT_EQ(s.type, UpdateStrategyType::REAR_WHEEL_UPDATE);
  EXPECT_NEAR(s.anchor_point.x, 8.0, 1e-6);  // rear face center of a [8,12]x[-1,1] box
  EXPECT_NEAR(s.anchor_point.y, 0.0, 1e-6);
}

// Rounded body (12-gon, 30 deg turns): no sharp corner -> hold (no corner, zero trust).
TEST(AnalyzePolygonGeometry, RoundedBodyHasNoCorner)
{
  std::vector<std::pair<double, double>> pts;
  for (int i = 0; i < 12; ++i) {
    const double th = 2.0 * M_PI * static_cast<double>(i) / 12.0;
    pts.push_back({10.0 + 1.5 * std::cos(th), 1.5 * std::sin(th)});
  }
  const auto cluster = makePolyCluster(pts);
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(0.0, -10.0), pred);

  EXPECT_FALSE(g.has_corner);
  EXPECT_DOUBLE_EQ(g.trust, 0.0);
}

// A non-polygon (bounding box, no footprint) yields no corner.
TEST(AnalyzePolygonGeometry, NonPolygonHasNoCorner)
{
  const auto cluster = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(0.0, -10.0), pred);

  EXPECT_FALSE(g.has_corner);
  EXPECT_DOUBLE_EQ(g.trust, 0.0);
}

// An over-wide hull (merge / over-segmentation): width far exceeds the track -> FAULTY inflation.
TEST(AnalyzePolygonGeometry, WideMergeFlagsFaulty)
{
  const auto cluster = makePolyCluster({{8, -2}, {12, -2}, {12, 2}, {8, 2}});  // width 4, area 16
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);                         // width 2, area 8
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(0.0, -10.0), pred);

  EXPECT_TRUE(g.has_corner);
  EXPECT_EQ(g.inflation, shapes::PolygonInflation::FAULTY);
}

// A thin rear cluster (only the rear face visible, no L-corner): the single visible end face is
// recovered with its midpoint as the end-face center (NOT a two-face corner).
TEST(AnalyzePolygonGeometry, ThinRearFaceRecoversEndFace)
{
  // Rear face at x=8 spanning y in [-1, 1], thin in x ([8, 8.2]); ego sits behind at x=0.
  const auto cluster = makePolyCluster({{8, -1}, {8.2, -1}, {8.2, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(0.0, 0.0), pred);

  EXPECT_FALSE(g.has_corner);
  EXPECT_TRUE(g.has_end_face);
  EXPECT_NEAR(g.end_face_center.x, 8.0, 1e-6);  // rear face center
  EXPECT_NEAR(g.end_face_center.y, 0.0, 1e-6);
  EXPECT_NEAR(g.observed_width, 2.0, 1e-6);
  EXPECT_GT(g.trust, 0.0);
}

// The strategy anchors the thin rear cluster at the REAR face (so the box extends forward), instead
// of degrading to a centroid-as-center weak update.
TEST(AnalyzePolygonGeometry, ThinRearFaceAnchorsAtRear)
{
  const auto cluster = makePolyCluster({{8, -1}, {8.2, -1}, {8.2, 1}, {8, 1}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(0.0, 0.0), pred);
  const auto s = determineUpdateStrategy(g, pred, 2.0);

  EXPECT_EQ(s.type, UpdateStrategyType::REAR_WHEEL_UPDATE);
  EXPECT_NEAR(s.anchor_point.x, 8.0, 1e-6);  // rear face center, not the box center
  EXPECT_NEAR(s.anchor_point.y, 0.0, 1e-6);
}

// A thin SIDE strip (visible edge parallel to the body axis) must NOT be treated as an end face:
// the longitudinal anchor is ambiguous, so it stays on the weak path.
TEST(AnalyzePolygonGeometry, ThinSideFaceStaysWeak)
{
  // Side face at y=1.2 spanning x in [8, 12], thin in y ([1, 1.2]); ego sits above at y=10.
  const auto cluster = makePolyCluster({{8, 1}, {12, 1}, {12, 1.2}, {8, 1.2}});
  const auto pred = makeBox(10.0, 0.0, 0.0, 4.0, 2.0);
  const auto g = shapes::analyzePolygonGeometry(cluster, makeEgo(10.0, 10.0), pred);
  const auto s = determineUpdateStrategy(g, pred, 2.0);

  EXPECT_FALSE(g.has_corner);
  EXPECT_FALSE(g.has_end_face);
  EXPECT_EQ(s.type, UpdateStrategyType::WEAK_UPDATE);
}

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

}  // namespace autoware::multi_object_tracker
