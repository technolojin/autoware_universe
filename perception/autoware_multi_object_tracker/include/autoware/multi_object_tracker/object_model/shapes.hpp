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

// Prior-driven corner measurement for the corner-based vehicle update.
//
// CONCEPT (the deliberate departure from "detect a corner in the cluster"): a tracked vehicle is a
// box with four corners. We do NOT try to classify whether the cluster shape "contains a corner" —
// real LiDAR clusters are not deterministic (rounded bumpers, sparse/occluded returns, mirrors,
// varying bend angles), so any shape-only "is this an L-corner?" test is unwinnable. Instead the
// PREDICTION asserts the box orientation and, with ego, which corner faces the sensor; the CLUSTER
// only measures WHERE that corner sits. The measured corner is the cluster's extreme point on the
// ego-facing longitudinal and lateral sides, read in the predicted-yaw box frame.
//
// Anti-feedback discipline (relaxed but bounded): the corner position is a pure function of the
// cluster points and the prior ORIENTATION only — the prior position / width / length never enter
// the mean. Borrowing the prior orientation to define the readout frame is the same linearization
// the EKF already performs at the predicted yaw; the corner constraint still pulls a drifted yaw
// back across frames. The covariance is observation-derived (per-face point support), never shrunk
// by agreement with the prediction, so genuine misalignment surfaces as an honest innovation.
struct PolygonMeasurement
{
  // Visible corner = the cluster's extreme point on the ego-facing longitudinal and lateral sides,
  // expressed in the predicted-yaw box frame and mapped back to the map frame. Rounding-immune: it
  // is an extent intersection, not a sampled apex. has_corner is true whenever at least one ego-
  // facing surface is resolved (the unobserved axis is handled by a large covariance below).
  bool has_corner = false;
  geometry_msgs::msg::Point corner;
  // 2x2 position covariance of `corner` [m^2], row-major {xx, xy, yx, yy}. Anisotropic by
  // construction: tight along an axis a real face resolves, large along an axis with no supporting
  // face (a single visible face leaves the corner nearly unconstrained perpendicular to it), so the
  // EKF only corrects the well-observed degrees of freedom and yaw becomes observable from the
  // well-localized side.
  std::array<double, 4> corner_cov{};

  // Which predicted box corner the measurement is associated to, derived from the ego-facing side
  // (a discrete prior-driven choice, not a mean injection). Consumed by VehicleTracker to
  // reconstruct the face-center measurement for the wheel-anchor EKF (updateStatePoseFront/Rear):
  // is_front selects the front/rear endpoint blend, s_lat (+1 / -1) the lateral half-width sign.
  bool is_front = false;
  double s_lat = 1.0;

  // One-sided lower bound on the body LENGTH from directly observed surface (occlusion only ever
  // shortens what is seen). For the separate grow-only length filter; must never shrink the tracked
  // length. Decoupled from the position measurement above. WIDTH is deliberately NOT reported: the
  // tracked width comes only from the bbox detector, never from a polygon cluster (a diagonal /
  // over-merged cluster would otherwise inflate width and offset the axles via the corner update's
  // lateral half-width term, destabilizing the EKF).
  double visible_length = 0.0;
};

// Prior-driven corner / extent extraction for the vehicle update (see PolygonMeasurement).
// `cluster` is a POLYGON DynamicObject (footprint in cluster-local frame, pose in map frame);
// `ego_pos` is the sensor/ego position in map frame (resolves which corner is visible);
// `prediction` supplies the box orientation that defines the readout frame and, with ego, the
// front/rear + left/right association — never a mean or a covariance. Returns has_corner=false only
// for a non-polygon / degenerate cluster or when no ego-facing surface is resolvable.
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
