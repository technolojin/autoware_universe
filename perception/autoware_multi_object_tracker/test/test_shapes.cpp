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

#include "autoware/multi_object_tracker/object_model/cluster_shape.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace autoware::multi_object_tracker
{
namespace
{
constexpr double kDeg = M_PI / 180.0;

// A POLYGON cluster whose footprint is built directly in the map frame (identity pose at origin,
// reference_yaw = 0 in the tests), so a footprint rotated by `yaw` represents a body misaligned
// from the read frame by exactly `yaw`. estimateFineYawCorrection should recover that `yaw`.
types::DynamicObject makeCluster()
{
  types::DynamicObject obj;
  obj.shape.type = autoware_perception_msgs::msg::Shape::POLYGON;
  obj.pose.orientation.w = 1.0;  // yaw 0
  return obj;
}

void addPoint(types::DynamicObject & obj, double x, double y)
{
  geometry_msgs::msg::Point32 p;
  p.x = static_cast<float>(x);
  p.y = static_cast<float>(y);
  p.z = 0.0F;
  obj.shape.footprint.points.push_back(p);
}

// Densely sampled rectangle outline of size length x width, rotated by `yaw` about the origin.
types::DynamicObject makeRect(double length, double width, double yaw, int n_per_edge = 24)
{
  auto obj = makeCluster();
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double hl = 0.5 * length, hw = 0.5 * width;
  auto add_local = [&](double lx, double ly) { addPoint(obj, c * lx - s * ly, s * lx + c * ly); };
  for (int i = 0; i <= n_per_edge; ++i) {
    const double t = -hl + length * i / n_per_edge;
    add_local(t, hw);   // top long edge
    add_local(t, -hw);  // bottom long edge
  }
  for (int i = 0; i <= n_per_edge; ++i) {
    const double t = -hw + width * i / n_per_edge;
    add_local(hl, t);   // front short edge
    add_local(-hl, t);  // rear short edge
  }
  return obj;
}

// A single densely sampled straight edge (a partial detection) of the given length, centered at the
// origin, oriented at `yaw` (segment running along the body-lateral axis, i.e. a front/rear face).
types::DynamicObject makeFrontFace(double width, double yaw, int n = 20)
{
  auto obj = makeCluster();
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double hw = 0.5 * width;
  for (int i = 0; i <= n; ++i) {
    const double ly = -hw + width * i / n;
    const double lx = 0.0;  // face at longitudinal 0
    addPoint(obj, c * lx - s * ly, s * lx + c * ly);
  }
  return obj;
}
}  // namespace

// Aligned rectangle: dominant long side is parallel to the read axis -> ~zero correction.
TEST(EstimateFineYawCorrection, AlignedRectIsZero)
{
  const auto r = shapes::estimateFineYawCorrection(makeRect(4.0, 1.8, 0.0), 0.0);
  ASSERT_TRUE(r.has_value());
  EXPECT_NEAR(*r, 0.0, 0.2 * kDeg);
}

// Small misalignment from the full L-shape: recovered from the long side.
TEST(EstimateFineYawCorrection, RecoversSmallYawFromLongSide)
{
  const auto r = shapes::estimateFineYawCorrection(makeRect(4.0, 1.8, 3.0 * kDeg), 0.0);
  ASSERT_TRUE(r.has_value());
  EXPECT_NEAR(*r, 3.0 * kDeg, 0.5 * kDeg);
}

// Partial detection: only the front face is visible. The lateral-axis edge still resolves yaw
// (angle folded mod 90 deg into the fine window).
TEST(EstimateFineYawCorrection, RecoversYawFromFrontFaceOnly)
{
  const auto r = shapes::estimateFineYawCorrection(makeFrontFace(1.8, 3.0 * kDeg), 0.0);
  ASSERT_TRUE(r.has_value());
  EXPECT_NEAR(*r, 3.0 * kDeg, 0.5 * kDeg);
}

// Beyond the fine window but below the reject threshold: clamped to +/- 5 deg.
TEST(EstimateFineYawCorrection, ClampsToFiveDegrees)
{
  const auto r = shapes::estimateFineYawCorrection(makeFrontFace(1.5, 7.0 * kDeg), 0.0);
  ASSERT_TRUE(r.has_value());
  EXPECT_NEAR(*r, 5.0 * kDeg, 1e-6);
}

// A large disagreement is an association problem, not a fine misalignment -> reject (nullopt).
TEST(EstimateFineYawCorrection, RejectsLargeDisagreement)
{
  const auto r = shapes::estimateFineYawCorrection(makeFrontFace(1.2, 12.0 * kDeg), 0.0);
  EXPECT_FALSE(r.has_value());
}

// No edge long enough to be a resolved face -> nullopt.
TEST(EstimateFineYawCorrection, RejectsTooShortEdge)
{
  const auto r = shapes::estimateFineYawCorrection(makeFrontFace(0.3, 2.0 * kDeg), 0.0);
  EXPECT_FALSE(r.has_value());
}

// A bounding box (non-polygon) carries no cluster footprint to fit -> nullopt.
TEST(EstimateFineYawCorrection, RejectsNonPolygon)
{
  auto obj = makeRect(4.0, 1.8, 3.0 * kDeg);
  obj.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  EXPECT_FALSE(shapes::estimateFineYawCorrection(obj, 0.0).has_value());
}

}  // namespace autoware::multi_object_tracker
