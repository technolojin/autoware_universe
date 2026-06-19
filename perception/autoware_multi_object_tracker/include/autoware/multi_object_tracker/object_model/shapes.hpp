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

// Classification of why an observed polygon is larger / differently shaped than the track.
enum class PolygonInflation {
  NONE,                   // observed extent consistent with the tracked shape
  NOISE,                  // a thin spike vertex (e.g. rain return) inflates the hull
  FAULTY,                 // bimodal / over-wide hull (merge or over-segmentation)
  SHAPE_CHANGE_CANDIDATE  // localized persistent extent growth (e.g. door opening); needs
                          // temporal confirmation before being trusted
};

// Geometry extracted from a single vehicle cluster polygon, decoupled from any filter state.
// A LiDAR vehicle cluster observes at most an L-shape: one long side + one end face meeting at the
// near corner; the far hull boundary is silhouette closure across the occluded region, not real
// surface. Only ego-facing edges are treated as real measurements. Confidences are continuous so
// the caller can scale measurement variance (and hold yaw when a cue is unreliable) rather than
// hard-accept / hard-reject.
struct PolygonGeometry
{
  // Near corner — the ego-facing junction of the two visible faces (the L corner), map frame.
  geometry_msgs::msg::Point near_corner;
  bool has_corner = false;  // true only when a genuine two-face corner was found

  // Orientation cue from the longer visible edge tangent (map-frame yaw, direction only — never
  // used as a length). yaw_variance is continuous: large for short / rounded / poorly supported
  // edges so the caller naturally holds yaw. yaw_cue_valid is the floor flag (length & straightness
  // & roughly parallel to the predicted body axis).
  double long_edge_dir = 0.0;
  double long_edge_len = 0.0;
  double yaw_variance = std::numeric_limits<double>::max();
  bool yaw_cue_valid = false;

  // Lateral extent of the visible side, perpendicular to long_edge_dir [m]; confidence in [0,1].
  // Intentionally decoupled from vehicle length.
  double observed_width = 0.0;
  double observed_width_confidence = 0.0;

  double roundness = 1.0;  // [0,1]; 1 = no straight edge resolvable
  PolygonInflation inflation = PolygonInflation::NONE;
  double trust = 0.0;  // overall [0,1] for the caller's path choice
};

// Analyze a vehicle cluster polygon into orientation / anchor cues with confidences.
// `cluster` is a POLYGON DynamicObject (footprint in cluster-local frame, pose in map frame);
// `ego_pos` is the sensor/ego position in map frame (required to resolve which surfaces are
// visible); `prediction` is the tracked object (map pose + bbox dims) used as the reference for
// 180-deg / front-rear disambiguation and width comparison. Returns a PolygonGeometry with
// has_corner=false / trust=0 when no reliable two-face corner can be extracted.
PolygonGeometry analyzePolygonGeometry(
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
