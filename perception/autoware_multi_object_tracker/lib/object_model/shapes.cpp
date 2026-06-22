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

#include "autoware/multi_object_tracker/object_model/shapes.hpp"

#include <Eigen/Geometry>
#include <autoware_utils_geometry/boost_geometry.hpp>
#include <autoware_utils_geometry/boost_polygon_utils.hpp>
#include <tf2/utils.hpp>

#include <autoware_perception_msgs/msg/shape.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <boost/geometry.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr double MIN_AREA = 1e-6;
constexpr double INVALID_SCORE = -1.0;

struct OrientedExtent
{
  double min_along, max_along, min_lat, max_lat;
};

// Project polygon footprint points onto a unit axis (cos_u, sin_u) and its perpendicular.
template <typename PointContainer>
inline OrientedExtent computeOrientedExtent(
  const PointContainer & points, const double cos_u, const double sin_u)
{
  OrientedExtent ext{
    std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
    std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
  for (const auto & p : points) {
    const double along = p.x * cos_u + p.y * sin_u;
    const double lat = -p.x * sin_u + p.y * cos_u;
    if (along < ext.min_along) ext.min_along = along;
    if (along > ext.max_along) ext.max_along = along;
    if (lat < ext.min_lat) ext.min_lat = lat;
    if (lat > ext.max_lat) ext.max_lat = lat;
  }
  return ext;
}
// --- helpers for estimateFineYawCorrection (A2 fine yaw adjustment, see shapes.hpp) -------------
// All physical / scene-independent: a real LiDAR face is flat to within point scatter and spans a
// length scale; below these thresholds the perpendicular angle is treated as unobserved.
constexpr double FINE_YAW_LIDAR_POINT_STD = 0.10;  // [m] cluster point scatter about a true surface
constexpr double FINE_YAW_SUPPORT_BAND =
  3.0 *
  FINE_YAW_LIDAR_POINT_STD;  // [m] a point supports a face within ~3 sigma of its extreme line
constexpr double FINE_YAW_MIN_FACE_EXTENT =
  0.5;  // [m] minimum tangential span of an edge to be a resolved face (not a tip / occlusion edge)
constexpr size_t FINE_YAW_MIN_SUPPORT_POINTS = 4;    // minimum supporting points for a stable PCA
constexpr double FINE_YAW_MAX = 5.0 * M_PI / 180.0;  // [rad] hard clamp on the correction
constexpr double FINE_YAW_PRE_CLAMP_REJECT =
  10.0 * M_PI / 180.0;  // [rad] beyond this it is not fine
constexpr double FINE_YAW_MAX_RESIDUAL_VAR =
  (2.0 * FINE_YAW_LIDAR_POINT_STD) * (2.0 * FINE_YAW_LIDAR_POINT_STD);  // [m^2] edge-flatness gate

// PCA of a support-point set expressed in the (along, lat) body frame. Returns the angle of its
// MAJOR axis relative to the along-axis (in (-pi/2, pi/2]) and the minor eigenvalue (perpendicular
// residual variance). For a flat edge the major axis lies along the edge tangent, so a body frame
// misaligned by delta yields a major-axis angle of delta (side face) or pi/2 + delta (front/rear).
struct EdgeFit
{
  double major_angle;
  double lambda_min;
};
inline EdgeFit fitEdgePca(const std::vector<std::pair<double, double>> & support)
{
  const double inv = 1.0 / static_cast<double>(support.size());
  double ma = 0.0, ml = 0.0;
  for (const auto & [a, l] : support) {
    ma += a;
    ml += l;
  }
  ma *= inv;
  ml *= inv;
  double s_aa = 0.0, s_al = 0.0, s_ll = 0.0;
  for (const auto & [a, l] : support) {
    const double da = a - ma, dl = l - ml;
    s_aa += da * da;
    s_al += da * dl;
    s_ll += dl * dl;
  }
  s_aa *= inv;
  s_al *= inv;
  s_ll *= inv;
  const double half_tr = 0.5 * (s_aa + s_ll);
  const double disc = std::sqrt(std::max(0.0, half_tr * half_tr - (s_aa * s_ll - s_al * s_al)));
  const double lambda_min = half_tr - disc;
  const double major_angle = 0.5 * std::atan2(2.0 * s_al, s_aa - s_ll);
  return {major_angle, lambda_min};
}

double getSumArea(const std::vector<autoware_utils_geometry::Polygon2d> & polygons)
{
  return std::accumulate(
    polygons.begin(), polygons.end(), 0.0, [](double acc, autoware_utils_geometry::Polygon2d p) {
      return acc + boost::geometry::area(p);
    });
}

double getIntersectionArea(
  const autoware_utils_geometry::Polygon2d & source_polygon,
  const autoware_utils_geometry::Polygon2d & target_polygon)
{
  std::vector<autoware_utils_geometry::Polygon2d> intersection_polygons;
  boost::geometry::intersection(source_polygon, target_polygon, intersection_polygons);
  return getSumArea(intersection_polygons);
}

double getUnionArea(
  const autoware_utils_geometry::Polygon2d & source_polygon,
  const autoware_utils_geometry::Polygon2d & target_polygon)
{
  std::vector<autoware_utils_geometry::Polygon2d> union_polygons;
  boost::geometry::union_(source_polygon, target_polygon, union_polygons);
  return getSumArea(union_polygons);
}

double getConvexShapeArea(
  const autoware_utils_geometry::Polygon2d & source_polygon,
  const autoware_utils_geometry::Polygon2d & target_polygon)
{
  boost::geometry::model::multi_polygon<autoware_utils_geometry::Polygon2d> union_polygons;
  boost::geometry::union_(source_polygon, target_polygon, union_polygons);

  autoware_utils_geometry::Polygon2d hull;
  boost::geometry::convex_hull(union_polygons, hull);
  return boost::geometry::area(hull);
}
}  // namespace

namespace autoware::multi_object_tracker
{
namespace shapes
{

double get1dIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object)
{
  constexpr double min_union_length = 0.1;  // As 0.01 used in 2dIoU, use 0.1 here
  constexpr double min_length = 1e-3;       // As 1e-6 used in 2dIoU, use 1e-3 here
  // Compute radii from dimensions (use max of x and y as diameter)
  const double r_src =
    std::max(source_object.shape.dimensions.x, source_object.shape.dimensions.y) * 0.5;
  const double r_tgt =
    std::max(target_object.shape.dimensions.x, target_object.shape.dimensions.y) * 0.5;
  // if radius is smaller than the minimum length, return 0.0
  if (r_src < min_length || r_tgt < min_length) return 0.0;
  // Ensure r1 is the larger radius
  const double r1 = std::max(r_tgt, r_src);
  const double r2 = std::min(r_tgt, r_src);
  const auto dx = source_object.pose.position.x - target_object.pose.position.x;
  const auto dy = source_object.pose.position.y - target_object.pose.position.y;
  // distance between centers
  const auto dist = std::sqrt(dx * dx + dy * dy);
  // if distance is larger than the sum of radius, return 0.0
  if (dist > r1 + r2 - min_length) return 0.0;
  // if distance is smaller than the difference of radius, return the ratio of the smaller radius to
  // the larger radius
  // Square used to mimic area ratio behavior as a rough 2D approximation
  if (dist < r1 - r2) return (r2 * r2) / (r1 * r1);
  // if distance is between the difference and the sum of radii, return the ratio of the
  // intersection length to the union length
  if (r1 + r2 + dist < min_union_length) return 0.0;
  const double intersection_length = r1 + r2 - dist;
  const double iou = intersection_length * r2 / (r1 * r1) * 0.5;
  return iou;
}

double get2dIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object,
  const double min_union_area)
{
  const auto source_polygon =
    autoware_utils_geometry::to_polygon2d(source_object.pose, source_object.shape);
  if (boost::geometry::area(source_polygon) < MIN_AREA) return 0.0;
  const auto target_polygon =
    autoware_utils_geometry::to_polygon2d(target_object.pose, target_object.shape);
  if (boost::geometry::area(target_polygon) < MIN_AREA) return 0.0;

  const double intersection_area = getIntersectionArea(source_polygon, target_polygon);
  if (intersection_area < MIN_AREA) return 0.0;
  const double union_area = getUnionArea(source_polygon, target_polygon);

  const double iou =
    union_area < min_union_area ? 0.0 : std::min(1.0, intersection_area / union_area);
  return iou;
}

double get2dGeneralizedIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object)
{
  const auto source_polygon =
    autoware_utils_geometry::to_polygon2d(source_object.pose, source_object.shape);
  const double source_area = boost::geometry::area(source_polygon);
  const auto target_polygon =
    autoware_utils_geometry::to_polygon2d(target_object.pose, target_object.shape);
  const double target_area = boost::geometry::area(target_polygon);
  if (source_area < MIN_AREA && target_area < MIN_AREA) return -1.0;

  const double intersection_area = getIntersectionArea(source_polygon, target_polygon);
  const double union_area = getUnionArea(source_polygon, target_polygon);
  const double iou = union_area < 0.01 ? 0.0 : std::min(1.0, intersection_area / union_area);
  const double convex_shape_area = getConvexShapeArea(source_polygon, target_polygon);

  return iou - (convex_shape_area - union_area) / convex_shape_area;
}

bool get2dPrecisionRecallGIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object,
  double & precision, double & recall, double & generalized_iou)
{
  const auto source_polygon =
    autoware_utils_geometry::to_polygon2d(source_object.pose, source_object.shape);
  const double source_area = boost::geometry::area(source_polygon);
  if (source_area < MIN_AREA) return false;
  const auto target_polygon =
    autoware_utils_geometry::to_polygon2d(target_object.pose, target_object.shape);
  const double target_area = boost::geometry::area(target_polygon);
  if (target_area < MIN_AREA) return false;

  const double intersection_area = getIntersectionArea(source_polygon, target_polygon);
  const double union_area = getUnionArea(source_polygon, target_polygon);
  const double convex_shape_area = getConvexShapeArea(source_polygon, target_polygon);
  const double iou = union_area < 0.01 ? 0.0 : std::min(1.0, intersection_area / union_area);

  precision = source_area < MIN_AREA ? 0.0 : std::min(1.0, intersection_area / source_area);
  recall = source_area < MIN_AREA ? 0.0 : std::min(1.0, intersection_area / target_area);
  generalized_iou = iou - (convex_shape_area - union_area) / convex_shape_area;

  return true;
}

bool convertConvexHullToBoundingBox(
  const types::DynamicObject & input_object, types::DynamicObject & output_object,
  const std::optional<geometry_msgs::msg::Point> & ego_pos)
{
  const auto & points = input_object.shape.footprint.points;
  if (points.size() < 3) {
    return false;
  }

  // Transform ego position into the object's local frame for the ego-facing edge filter.
  // Footprint points are defined in local frame (object-relative 2D coords).
  double ego_local_x = 0.0, ego_local_y = 0.0;
  const bool use_ego = ego_pos.has_value();
  if (use_ego) {
    const double yaw = tf2::getYaw(input_object.pose.orientation);
    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double dgx = ego_pos->x - input_object.pose.position.x;
    const double dgy = ego_pos->y - input_object.pose.position.y;
    ego_local_x = cos_yaw * dgx + sin_yaw * dgy;
    ego_local_y = -sin_yaw * dgx + cos_yaw * dgy;
  }

  const size_t n = points.size();
  double best_area = std::numeric_limits<double>::max();
  size_t best_i = 0;
  OrientedExtent best_ext{};
  bool found_any = false;

  auto tryEdge = [&](const size_t i) -> bool {
    const auto & p0 = points[i];
    const auto & p1 = points[(i + 1) % n];
    const double ex = p1.x - p0.x, ey = p1.y - p0.y;
    const double len_sq = ex * ex + ey * ey;
    if (len_sq < 1e-12) return false;
    const double edge_len = std::sqrt(len_sq);
    const auto ext = computeOrientedExtent(points, ex / edge_len, ey / edge_len);
    const double area = (ext.max_along - ext.min_along) * (ext.max_lat - ext.min_lat);
    if (area < best_area) {
      best_area = area;
      best_i = i;
      best_ext = ext;
      return true;
    }
    return false;
  };

  // Ego-facing pass: outward normal of CCW edge (ex,ey) is (ey,-ex).
  // Edge faces ego when (ey,-ex)·(ego_local_x,ego_local_y) > 0, i.e. ey*ego_local_x -
  // ex*ego_local_y > 0.
  if (use_ego) {
    for (size_t i = 0; i < n; ++i) {
      const auto & p0 = points[i];
      const auto & p1 = points[(i + 1) % n];
      const double ex = p1.x - p0.x;
      const double ey = p1.y - p0.y;
      if (ey * ego_local_x - ex * ego_local_y <= 0.0) continue;
      found_any |= tryEdge(i);
    }
  }

  // Fallback: full per-edge search (no ego, CW polygon, or no ego-facing edges).
  if (!found_any) {
    for (size_t i = 0; i < n; ++i) {
      found_any |= tryEdge(i);
    }
    if (!found_any) return false;
  }

  // Recover edge direction and bbox geometry from the winning edge.
  const auto & p0 = points[best_i];
  const auto & p1 = points[(best_i + 1) % n];
  const double ex = p1.x - p0.x, ey = p1.y - p0.y;
  const double edge_len = std::sqrt(ex * ex + ey * ey);
  const double cos_u = ex / edge_len, sin_u = ey / edge_len;

  const double dim_along = best_ext.max_along - best_ext.min_along;
  const double dim_perp = best_ext.max_lat - best_ext.min_lat;

  // Bbox center in local frame: inverse of the normalized projection.
  const double cu = (best_ext.min_along + best_ext.max_along) * 0.5;
  const double cv = (best_ext.min_lat + best_ext.max_lat) * 0.5;
  const double center_local_x = cu * cos_u - cv * sin_u;
  const double center_local_y = cu * sin_u + cv * cos_u;

  const double bbox_yaw_local = std::atan2(ey, ex);

  const double obj_yaw = tf2::getYaw(input_object.pose.orientation);
  const double cos_yaw = std::cos(obj_yaw);
  const double sin_yaw = std::sin(obj_yaw);

  output_object = input_object;

  // Rotate local center offset to global and add to object position
  output_object.pose.position.x += cos_yaw * center_local_x - sin_yaw * center_local_y;
  output_object.pose.position.y += sin_yaw * center_local_x + cos_yaw * center_local_y;

  // Set global bbox orientation (object yaw + edge yaw in local frame)
  const double half = (obj_yaw + bbox_yaw_local) * 0.5;
  output_object.pose.orientation.x = 0.0;
  output_object.pose.orientation.y = 0.0;
  output_object.pose.orientation.z = std::sin(half);
  output_object.pose.orientation.w = std::cos(half);

  output_object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  output_object.shape.dimensions.x = dim_along;
  output_object.shape.dimensions.y = dim_perp;

  // Shift footprint to new center and rotate into new object frame (which is rotated by
  // bbox_yaw_local)
  for (auto & point : output_object.shape.footprint.points) {
    const float dx = point.x - static_cast<float>(center_local_x);
    const float dy = point.y - static_cast<float>(center_local_y);
    point.x = static_cast<float>(cos_u) * dx + static_cast<float>(sin_u) * dy;
    point.y = -static_cast<float>(sin_u) * dx + static_cast<float>(cos_u) * dy;
  }

  return true;
}

std::optional<types::DynamicObject> alignClusterToOrientation(
  const types::DynamicObject & cluster, const double target_yaw)
{
  if (
    cluster.shape.type != autoware_perception_msgs::msg::Shape::POLYGON ||
    cluster.shape.footprint.points.empty()) {
    return std::nullopt;
  }

  // Compose the two rotations (cluster local → map, map → target frame) into one.
  const double phi = target_yaw - tf2::getYaw(cluster.pose.orientation);
  const auto ext =
    computeOrientedExtent(cluster.shape.footprint.points, std::cos(phi), std::sin(phi));

  const double long_center = (ext.min_along + ext.max_along) * 0.5;
  const double lat_center = (ext.min_lat + ext.max_lat) * 0.5;
  const double cos_tr = std::cos(target_yaw), sin_tr = std::sin(target_yaw);

  types::DynamicObject aligned = cluster;
  aligned.pose.position.x = cluster.pose.position.x + long_center * cos_tr - lat_center * sin_tr;
  aligned.pose.position.y = cluster.pose.position.y + long_center * sin_tr + lat_center * cos_tr;

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, target_yaw);
  aligned.pose.orientation = tf2::toMsg(q);

  aligned.shape.dimensions.x = ext.max_along - ext.min_along;
  aligned.shape.dimensions.y = ext.max_lat - ext.min_lat;

  return aligned;
}

std::optional<double> estimateFineYawCorrection(
  const types::DynamicObject & cluster, const double reference_yaw)
{
  const auto & pts = cluster.shape.footprint.points;
  if (cluster.shape.type != autoware_perception_msgs::msg::Shape::POLYGON || pts.size() < 3) {
    return std::nullopt;
  }

  // Project every cluster-local point into the body frame at reference_yaw. phi composes the
  // cluster-local -> map -> body rotations into one angle (same convention as
  // computeOrientedExtent), so (a, l) are body-longitudinal / body-lateral coordinates and a flat
  // edge's PCA major axis angle is the residual misalignment directly.
  const double phi = reference_yaw - tf2::getYaw(cluster.pose.orientation);
  const double cphi = std::cos(phi), sphi = std::sin(phi);
  std::vector<std::pair<double, double>> al;
  al.reserve(pts.size());
  const auto ext = computeOrientedExtent(pts, cphi, sphi);
  for (const auto & p : pts) {
    al.emplace_back(p.x * cphi + p.y * sphi, -p.x * sphi + p.y * cphi);
  }

  // Four candidate faces (the two longitudinal extremes and the two lateral extremes). Each is
  // resolved by the points within FINE_YAW_SUPPORT_BAND of its extreme line; its TANGENTIAL span is
  // measured along the other axis. The face with the largest span is the longest visible straight
  // edge — the most reliable yaw lever (a long side beats a short, noisy front/rear).
  struct Face
  {
    double extreme;      // extreme value on the perpendicular axis
    bool perp_is_along;  // true: extreme is on the along-axis (front/rear), tangential = lat
  };
  const std::array<Face, 4> faces = {
    Face{ext.min_along, true}, Face{ext.max_along, true}, Face{ext.min_lat, false},
    Face{ext.max_lat, false}};

  double best_span = 0.0;
  std::vector<std::pair<double, double>> best_support;
  for (const auto & f : faces) {
    // Tangential extremes mark where the two adjacent (perpendicular) faces meet this one — the
    // corners. Fit the flat INTERIOR only: a corner blends in the perpendicular face and biases the
    // tangent (an L-shape otherwise pulls the PCA major axis toward the diagonal).
    const double t_min = f.perp_is_along ? ext.min_lat : ext.min_along;
    const double t_max = f.perp_is_along ? ext.max_lat : ext.max_along;
    std::vector<std::pair<double, double>> support;
    double tmin = std::numeric_limits<double>::max(), tmax = std::numeric_limits<double>::lowest();
    for (const auto & [a, l] : al) {
      const double perp = f.perp_is_along ? a : l;
      if (std::abs(perp - f.extreme) > FINE_YAW_SUPPORT_BAND) continue;
      const double tang = f.perp_is_along ? l : a;
      if (tang - t_min < FINE_YAW_SUPPORT_BAND || t_max - tang < FINE_YAW_SUPPORT_BAND) continue;
      tmin = std::min(tmin, tang);
      tmax = std::max(tmax, tang);
      support.emplace_back(a, l);
    }
    const double span = tmax - tmin;
    if (support.size() >= FINE_YAW_MIN_SUPPORT_POINTS && span > best_span) {
      best_span = span;
      best_support = std::move(support);
    }
  }

  // No edge long enough to be a resolved face -> not observable, leave yaw to the motion model.
  if (best_span < FINE_YAW_MIN_FACE_EXTENT) return std::nullopt;

  const EdgeFit fit = fitEdgePca(best_support);

  // Edge must actually be flat: a large perpendicular residual means a curved / noisy boundary, not
  // a true surface, so its tangent angle is not a trustworthy yaw cue.
  if (fit.lambda_min > FINE_YAW_MAX_RESIDUAL_VAR) return std::nullopt;

  // Fold the major-axis angle into (-45 deg, 45 deg]: the edge tangent may be the along-axis (side
  // face, angle ~ delta) or the lat-axis (front/rear, angle ~ 90 deg + delta); both reduce to the
  // same small misalignment delta. delta > 0 means rotating the body frame by +delta aligns it.
  double delta = fit.major_angle;
  delta -= M_PI_2 * std::round(delta / M_PI_2);

  // A large disagreement is an association / wrong-edge problem, not a fine misalignment: reject
  // and fall back rather than snapping the readout frame by a big, unverified angle.
  if (std::abs(delta) > FINE_YAW_PRE_CLAMP_REJECT) return std::nullopt;

  return std::clamp(delta, -FINE_YAW_MAX, FINE_YAW_MAX);
}

std::pair<double, double> getObjectZRange(const types::DynamicObject & object)
{
  const double center_z = object.pose.position.z;
  const double height = object.shape.dimensions.z;
  const double min_z = center_z - height / 2.0;
  const double max_z = center_z + height / 2.0;
  return {min_z, max_z};
}

double get3dGeneralizedIoU(
  const types::DynamicObject & source_object, const types::DynamicObject & target_object)
{
  const auto source_polygon =
    autoware_utils_geometry::to_polygon2d(source_object.pose, source_object.shape);
  if (boost::geometry::area(source_polygon) < MIN_AREA) return INVALID_SCORE;
  const auto target_polygon =
    autoware_utils_geometry::to_polygon2d(target_object.pose, target_object.shape);
  if (boost::geometry::area(target_polygon) < MIN_AREA) return INVALID_SCORE;

  const double union_area = getUnionArea(source_polygon, target_polygon);
  if (union_area < MIN_AREA) return INVALID_SCORE;

  const double intersection_area = getIntersectionArea(source_polygon, target_polygon);
  const double convex_area = getConvexShapeArea(source_polygon, target_polygon);

  const auto [z_min_src, z_max_src] = getObjectZRange(source_object);
  const auto [z_min_tgt, z_max_tgt] = getObjectZRange(target_object);

  const double height_overlap =
    std::max(0.0, std::min(z_max_src, z_max_tgt) - std::max(z_min_src, z_min_tgt));

  if (height_overlap <= 0.0) return INVALID_SCORE;

  const double total_height = std::max(z_max_src, z_max_tgt) - std::min(z_min_src, z_min_tgt);

  const double iou =
    std::clamp((intersection_area * height_overlap) / (union_area * total_height), 0.0, 1.0);

  return iou - (convex_area - union_area) / convex_area;
}

geometry_msgs::msg::Polygon unionFootprints(
  const geometry_msgs::msg::Polygon & a, const geometry_msgs::msg::Polygon & b)
{
  if (a.points.empty()) return b;
  if (b.points.empty()) return a;

  const auto to_boost = [](const geometry_msgs::msg::Polygon & fp) {
    autoware_utils_geometry::Polygon2d poly;
    for (const auto & p : fp.points) {
      poly.outer().emplace_back(p.x, p.y);
    }
    boost::geometry::correct(poly);
    return poly;
  };

  // Extract exterior ring into msg::Polygon; skip Boost's closing duplicate point.
  const auto to_msg = [](const autoware_utils_geometry::Polygon2d & poly) {
    geometry_msgs::msg::Polygon out;
    const auto & ring = poly.outer();
    const size_t n = ring.size() > 1u ? ring.size() - 1u : ring.size();
    out.points.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      geometry_msgs::msg::Point32 p;
      p.x = static_cast<float>(ring[i].x());
      p.y = static_cast<float>(ring[i].y());
      p.z = 0.0f;
      out.points.push_back(p);
    }
    return out;
  };

  const auto poly_a = to_boost(a);
  const auto poly_b = to_boost(b);

  std::vector<autoware_utils_geometry::Polygon2d> union_result;
  boost::geometry::union_(poly_a, poly_b, union_result);
  if (union_result.empty()) return a;

  // Single connected result — extract directly.
  if (union_result.size() == 1u) {
    return to_msg(union_result[0]);
  }

  // Disjoint components: compute convex hull of all component vertices so that both
  // footprints are covered without discarding the smaller one.
  autoware_utils_geometry::Polygon2d all_points;
  for (const auto & comp : union_result) {
    for (const auto & pt : comp.outer()) {
      all_points.outer().push_back(pt);
    }
  }
  autoware_utils_geometry::Polygon2d hull;
  boost::geometry::convex_hull(all_points, hull);
  return to_msg(hull);
}

geometry_msgs::msg::Polygon transformFootprint(
  const geometry_msgs::msg::Polygon & footprint, const geometry_msgs::msg::Pose & src_pose,
  const geometry_msgs::msg::Pose & dst_pose)
{
  const double src_yaw = tf2::getYaw(src_pose.orientation);
  const double dst_yaw = tf2::getYaw(dst_pose.orientation);
  const double d_yaw = src_yaw - dst_yaw;
  const double cos_d = std::cos(d_yaw);
  const double sin_d = std::sin(d_yaw);
  const double cos_dst = std::cos(dst_yaw);
  const double sin_dst = std::sin(dst_yaw);
  const double wx = src_pose.position.x - dst_pose.position.x;
  const double wy = src_pose.position.y - dst_pose.position.y;
  const double t_x = cos_dst * wx + sin_dst * wy;
  const double t_y = -sin_dst * wx + cos_dst * wy;

  geometry_msgs::msg::Polygon result;
  result.points.resize(footprint.points.size());
  for (size_t i = 0; i < footprint.points.size(); ++i) {
    const auto & p = footprint.points[i];
    result.points[i].x = static_cast<float>(cos_d * p.x - sin_d * p.y + t_x);
    result.points[i].y = static_cast<float>(sin_d * p.x + cos_d * p.y + t_y);
    result.points[i].z = p.z;
  }
  return result;
}

}  // namespace shapes

}  // namespace autoware::multi_object_tracker
