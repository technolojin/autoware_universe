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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__OBJECT_MODEL__CLUSTER_SHAPE_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__OBJECT_MODEL__CLUSTER_SHAPE_HPP_

// Cluster / polygon shape geometry used by the vehicle (and pedestrian) tracker update: turning a
// raw LiDAR cluster footprint into body-frame extents, an oriented bounding box, or a fine heading
// correction. Kept separate from shapes.hpp, which holds the IoU / overlap scoring used for
// association.

#include "autoware/multi_object_tracker/types.hpp"

#include <geometry_msgs/msg/point.hpp>

#include <optional>

namespace autoware::multi_object_tracker
{
namespace shapes
{

// Fit a minimum-area oriented bounding box to a convex-hull / polygon cluster and write it into
// output_object (as a BOUNDING_BOX). When `ego_pos` is given, the search is restricted to the
// ego-facing edges (the observed surfaces), falling back to a full per-edge search otherwise.
bool convertConvexHullToBoundingBox(
  const types::DynamicObject & input_object, types::DynamicObject & output_object,
  const std::optional<geometry_msgs::msg::Point> & ego_pos = std::nullopt);

// Re-express a polygon cluster as an axis-aligned bounding box in the `target_yaw` frame: the
// returned object carries the cluster's oriented extents (dimensions) and the extent-center pose at
// `target_yaw`. Returns nullopt for a non-polygon / empty footprint.
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

}  // namespace shapes
}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__OBJECT_MODEL__CLUSTER_SHAPE_HPP_
