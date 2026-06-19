// Copyright 2024 TIER IV, Inc.
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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__OBJECT_MODEL__SHAPES_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__OBJECT_MODEL__SHAPES_HPP_

#include "autoware/multi_object_tracker/types.hpp"

#include <geometry_msgs/msg/point.hpp>

#include <array>
#include <limits>
#include <optional>
#include <utility>

namespace autoware::multi_object_tracker
{
namespace shapes
{

double get1dIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object);

double get2dIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object,
  const double min_union_area = 0.01);

double get2dGeneralizedIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object);

bool get2dPrecisionRecallGIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object,
  double & precision, double & recall, double & generalized_iou);

bool convertConvexHullToBoundingBox(
  const types::DynamicObject & input_object, types::DynamicObject & output_object,
  const std::optional<geometry_msgs::msg::Point> & ego_pos = std::nullopt);

std::optional<types::DynamicObject> alignClusterToOrientation(
  const types::DynamicObject & cluster, double target_yaw);

// Observation-only polygon measurement for the corner-based vehicle update. Every field is a pure
// function of the cluster footprint and the sensor geometry — NO prior (tracked center / yaw /
// width / length) enters the mean OR the covariance. The prior is consumed strictly downstream as
// (a) a discrete face association and (b) the linear measurement model inside the EKF, never
// injected here. Keeping the mean observation-only and the covariance derived from the measured
// point geometry (not from agreement with the prediction) is what prevents a self-confirming
// measurement -> EKF -> measurement feedback loop, and lets genuine misalignment (motion lag,
// shape change, mis-clustering) surface as an honest innovation instead of being masked.
struct PolygonMeasurement
{
  // Near corner = intersection of the two fitted visible faces (rounding-immune: it is the meeting
  // point of the two surface lines, not a sampled vertex). Map frame. has_corner is true only when
  // two statistically distinct ego-facing faces are resolved.
  bool has_corner = false;
  geometry_msgs::msg::Point corner;
  // 2x2 position covariance of `corner` [m^2], row-major {xx, xy, yx, yy}, propagated from the two
  // line fits. Naturally anisotropic: large along the ill-determined axis as the two faces approach
  // parallel, so the EKF only corrects the well-observed degrees of freedom.
  std::array<double, 4> corner_cov{};

  // Heading cue from the longer visible face tangent (direction only — never a length). yaw_var is
  // continuous, from the point noise and the edge spread, so short / poorly supported edges
  // naturally carry near-zero weight. Resolved against the predicted axis only for the 180-deg
  // branch (a discrete choice, not a mean injection).
  bool has_yaw = false;
  double yaw = 0.0;
  double yaw_var = std::numeric_limits<double>::max();

  // One-sided lower bounds on the body dimensions from directly observed surface (occlusion only
  // ever shortens what is seen). For the separate dimension filter; must never shrink a tracked
  // dimension. Decoupled from the position measurement above.
  double visible_length = 0.0;
  double visible_width = 0.0;
};

// Observation-only corner / orientation extraction for the vehicle update (see PolygonMeasurement).
// `cluster` is a POLYGON DynamicObject (footprint in cluster-local frame, pose in map frame);
// `ego_pos` is the sensor/ego position in map frame (resolves which faces are visible);
// `prediction` is used ONLY to disambiguate the 180-deg yaw branch — never to build a mean or a
// covariance. Returns has_corner=false when fewer than two statistically distinct ego-facing faces
// are present.
PolygonMeasurement analyzePolygonMeasurement(
  const types::DynamicObject & cluster, const geometry_msgs::msg::Point & ego_pos,
  const types::DynamicObject & prediction);

std::pair<double, double> getObjectZRange(const types::DynamicObject & object);

double get3dGeneralizedIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object);

// Transform polygon footprint points from src_pose's local frame into dst_pose's local frame.
// Equivalent to: p_dst = R_dst^T * (R_src * p_src + t_src - t_dst)
geometry_msgs::msg::Polygon transformFootprint(
  const geometry_msgs::msg::Polygon & footprint, const geometry_msgs::msg::Pose & src_pose,
  const geometry_msgs::msg::Pose & dst_pose);

// Compute the polygon union of two footprints already expressed in the same local frame.
// Returns the exterior ring of the union polygon, or the convex hull of all vertices when
// the inputs are disjoint.
geometry_msgs::msg::Polygon unionFootprints(
  const geometry_msgs::msg::Polygon & a, const geometry_msgs::msg::Polygon & b);

}  // namespace shapes
}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__OBJECT_MODEL__SHAPES_HPP_
