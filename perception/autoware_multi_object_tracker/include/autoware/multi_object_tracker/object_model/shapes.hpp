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

// Fine yaw adjustment (A2): the small heading correction that best aligns the cluster's dominant
// visible flat edge to the body axes, found by a least-squares (PCA) fit of that edge's supporting
// points and folded into a +/- 5 deg window. This is a READOUT aid only — the caller applies it to
// the cluster-alignment frame (alignClusterToOrientation), NEVER to the EKF yaw state, so it
// carries zero feedback into the filter: it merely de-biases WHERE the partial polygon is read.
//
// `reference_yaw` is the tracked heading the cluster is currently read against. Returns nullopt
// when no edge is observable enough to trust (too short / too few points / not flat) or when the
// raw correction exceeds the "fine" window (a large disagreement is an association problem, not a
// small misalignment, and is left to the motion model). The returned value is in radians, |delta|
// <= ~5 deg.
std::optional<double> estimateFineYawCorrection(
  const types::DynamicObject & cluster, double reference_yaw);

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
