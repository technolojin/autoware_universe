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
#include <cmath>
#include <cstdint>
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

namespace
{
// --- helpers for analyzePolygonGeometry -------------------------------------------------------

struct Vec2
{
  double x{0.0};
  double y{0.0};
};

inline double normAngle(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

// Footprint points (cluster-local) -> map frame, using the cluster pose.
std::vector<Vec2> footprintToMap(const types::DynamicObject & cluster)
{
  const double yaw = tf2::getYaw(cluster.pose.orientation);
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double tx = cluster.pose.position.x, ty = cluster.pose.position.y;
  std::vector<Vec2> out;
  out.reserve(cluster.shape.footprint.points.size());
  for (const auto & p : cluster.shape.footprint.points) {
    out.push_back({tx + c * p.x - s * p.y, ty + s * p.x + c * p.y});
  }
  return out;
}

double signedArea(const std::vector<Vec2> & p)
{
  double a = 0.0;
  const size_t n = p.size();
  for (size_t i = 0; i < n; ++i) {
    const auto & q0 = p[i];
    const auto & q1 = p[(i + 1) % n];
    a += q0.x * q1.y - q1.x * q0.y;
  }
  return 0.5 * a;
}

// Total-least-squares line fit over a vertex subset: principal direction, perpendicular RMS
// residual, and the extent (length) projected along that direction.
struct LineFit
{
  double dir = 0.0;       // [rad]
  double residual = 0.0;  // perpendicular RMS [m]
  double length = 0.0;    // extent along dir [m]
  bool valid = false;
};

LineFit fitLine(const std::vector<Vec2> & pts)
{
  LineFit f;
  const size_t n = pts.size();
  if (n < 2) return f;
  double cx = 0.0, cy = 0.0;
  for (const auto & p : pts) {
    cx += p.x;
    cy += p.y;
  }
  cx /= static_cast<double>(n);
  cy /= static_cast<double>(n);
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (const auto & p : pts) {
    const double dx = p.x - cx, dy = p.y - cy;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  f.dir = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
  const double cd = std::cos(f.dir), sd = std::sin(f.dir);
  double min_along = std::numeric_limits<double>::max(),
         max_along = std::numeric_limits<double>::lowest();
  double sq_perp = 0.0;
  for (const auto & p : pts) {
    const double dx = p.x - cx, dy = p.y - cy;
    const double along = dx * cd + dy * sd;
    const double perp = -dx * sd + dy * cd;
    min_along = std::min(min_along, along);
    max_along = std::max(max_along, along);
    sq_perp += perp * perp;
  }
  f.length = max_along - min_along;
  f.residual = std::sqrt(sq_perp / static_cast<double>(n));
  f.valid = true;
  return f;
}
}  // namespace

PolygonGeometry analyzePolygonGeometry(
  const types::DynamicObject & cluster, const geometry_msgs::msg::Point & ego_pos,
  const types::DynamicObject & prediction)
{
  // Tunables (documented inline; kept local — this is a self-contained geometry analyzer).
  constexpr double MIN_ARM_LEN = 0.5;                // [m] min length of each visible face arm
  constexpr double TURN_MIN = 35.0 * M_PI / 180.0;   // corner must bend at least this much
  constexpr double TURN_MAX = 150.0 * M_PI / 180.0;  // ...and not fully reverse (spike guard)
  constexpr double LONG_EDGE_MIN = 1.0;  // [m] min long-edge length to trust its tangent
  constexpr double STRAIGHT_MIN = 0.5;   // straightness floor for yaw_cue_valid
  constexpr double ANGLE_MAX_TO_PRED = 45.0 * M_PI / 180.0;  // long edge ~parallel to body axis
  constexpr double R_REF = 0.15;                             // [m] residual scale for straightness
  constexpr double BASE_YAW_VAR = (5.0 * M_PI / 180.0) * (5.0 * M_PI / 180.0);  // [rad^2]
  constexpr double REF_LEN = 3.0;              // [m] reference long-edge length
  constexpr double S_FLOOR = 0.1;              // straightness floor inside variance scaling
  constexpr double AREA_RATIO_INFLATE = 1.5;   // area mismatch beyond this flags inflation
  constexpr double WIDTH_INFLATE_RATIO = 1.5;  // observed width beyond this * tracked -> FAULTY
  constexpr double SPIKE_TURN = 150.0 * M_PI / 180.0;  // thin-spike vertex turn (noise)
  // Single visible end-face fallback (thin rear/front cluster: one face, no L-corner).
  constexpr double END_FACE_PERP_TOL = 40.0 * M_PI / 180.0;  // edge ~perpendicular to body axis
  constexpr double END_FACE_MIN_LEN = 0.3;       // [m] min visible edge length to anchor
  constexpr double END_FACE_WIDTH_MARGIN = 1.5;  // edge length must be <= margin * tracked width
  constexpr double END_FACE_MAX_RESIDUAL = 0.3;  // [m] max RMS straightness residual of the edge
  constexpr double END_FACE_TRUST = 0.4;         // conservative trust for the single-face anchor

  PolygonGeometry g;

  const auto & raw = cluster.shape.footprint.points;
  if (cluster.shape.type != autoware_perception_msgs::msg::Shape::POLYGON || raw.size() < 3) {
    return g;  // not a usable polygon -> trust 0
  }

  std::vector<Vec2> pts = footprintToMap(cluster);
  const size_t n = pts.size();
  if (signedArea(pts) < 0.0) std::reverse(pts.begin(), pts.end());  // enforce CCW

  const double pred_yaw = tf2::getYaw(prediction.pose.orientation);
  const double pred_width = std::max(0.1, prediction.shape.dimensions.y);

  // Ego-facing edge mask. For a CCW polygon the outward normal of edge (e_x,e_y) is (e_y,-e_x);
  // the edge faces ego when that normal points toward ego from the edge midpoint.
  std::vector<bool> facing(n, false);
  for (size_t i = 0; i < n; ++i) {
    const auto & p0 = pts[i];
    const auto & p1 = pts[(i + 1) % n];
    const double ex = p1.x - p0.x, ey = p1.y - p0.y;
    const double mx = 0.5 * (p0.x + p1.x), my = 0.5 * (p0.y + p1.y);
    const double to_ego_x = ego_pos.x - mx, to_ego_y = ego_pos.y - my;
    facing[i] = (ey * to_ego_x - ex * to_ego_y) > 0.0;
  }

  // Single visible end-face fallback. When no two-face corner is usable (rounded body, or only one
  // real face — a thin rear/front cluster), the longest contiguous ego-facing edge run IS the
  // visible end face: its midpoint is the observed end-face center. Accept it only when that edge
  // runs ~perpendicular to the predicted body axis (a genuine front/rear face, not a long side)
  // and its length is plausible against the tracked width. Populates the end-face cue on success;
  // leaves g untouched (trust 0) otherwise, so the caller falls back to the weak blended update.
  const double z_cluster = cluster.pose.position.z;
  auto attemptEndFace = [&]() {
    // Longest contiguous (cyclic) run of ego-facing edges. A fully ego-facing hull is degenerate
    // (faces indistinguishable) and is rejected.
    bool all_facing = true;
    for (size_t i = 0; i < n; ++i) {
      if (!facing[i]) {
        all_facing = false;
        break;
      }
    }
    if (all_facing) return;
    size_t break_edge = 0;
    while (break_edge < n && facing[break_edge]) ++break_edge;
    size_t best_start = 0, best_run = 0, cur_start = 0, cur_run = 0;
    for (size_t step = 0; step < n; ++step) {
      const size_t e = (break_edge + 1 + step) % n;  // start scanning just after a non-facing edge
      if (facing[e]) {
        if (cur_run == 0) cur_start = e;
        ++cur_run;
        if (cur_run > best_run) {
          best_run = cur_run;
          best_start = cur_start;
        }
      } else {
        cur_run = 0;
      }
    }
    if (best_run < 1) return;

    // Vertices spanned by the run (best_run edges -> best_run + 1 vertices).
    std::vector<Vec2> run;
    run.reserve(best_run + 1);
    for (size_t t = 0; t <= best_run; ++t) run.push_back(pts[(best_start + t) % n]);

    const LineFit lf = fitLine(run);
    if (!lf.valid || lf.length < END_FACE_MIN_LEN || lf.residual > END_FACE_MAX_RESIDUAL) return;
    // End-face gate: edge ~perpendicular to the predicted body axis.
    const double off = std::abs(normAngle(lf.dir - pred_yaw));
    if (std::abs(off - M_PI_2) > END_FACE_PERP_TOL) return;
    // Width plausibility: the visible face cannot be much wider than the tracked width.
    if (lf.length > pred_width * END_FACE_WIDTH_MARGIN) return;

    // End-face center = projected midpoint of the run along the edge tangent.
    double cx = 0.0, cy = 0.0;
    for (const auto & v : run) {
      cx += v.x;
      cy += v.y;
    }
    cx /= static_cast<double>(run.size());
    cy /= static_cast<double>(run.size());
    const double cd = std::cos(lf.dir), sd = std::sin(lf.dir);
    double min_a = std::numeric_limits<double>::max(), max_a = std::numeric_limits<double>::lowest();
    for (const auto & v : run) {
      const double a = (v.x - cx) * cd + (v.y - cy) * sd;
      min_a = std::min(min_a, a);
      max_a = std::max(max_a, a);
    }
    const double mid = 0.5 * (min_a + max_a);

    g.has_corner = false;
    g.has_end_face = true;
    g.end_face_center.x = cx + mid * cd;
    g.end_face_center.y = cy + mid * sd;
    g.end_face_center.z = z_cluster;
    g.long_edge_dir = lf.dir;  // edge tangent = body lateral axis in this mode
    g.observed_width = lf.length;
    g.yaw_cue_valid = false;
    g.trust = END_FACE_TRUST;
  };

  // Find the corner: an ego-facing vertex whose two incident edges both face ego and bend by a
  // near-right angle. Among candidates, pick the one closest to ego (the most reliable corner).
  auto edgeDir = [&](size_t i) {
    const auto & a = pts[i];
    const auto & b = pts[(i + 1) % n];
    return std::atan2(b.y - a.y, b.x - a.x);
  };
  int64_t best_corner = -1;
  double best_corner_range = std::numeric_limits<double>::max();
  double max_vertex_turn = 0.0;  // for spike (noise) detection, over all vertices
  for (size_t j = 0; j < n; ++j) {
    const size_t e_in = (j + n - 1) % n;  // edge (j-1, j)
    const size_t e_out = j;               // edge (j, j+1)
    const double turn = std::abs(normAngle(edgeDir(e_out) - edgeDir(e_in)));
    max_vertex_turn = std::max(max_vertex_turn, turn);
    if (!facing[e_in] || !facing[e_out]) continue;
    if (turn < TURN_MIN || turn > TURN_MAX) continue;
    const double dx = pts[j].x - ego_pos.x, dy = pts[j].y - ego_pos.y;
    const double range = std::hypot(dx, dy);
    if (range < best_corner_range) {
      best_corner_range = range;
      best_corner = static_cast<int64_t>(j);
    }
  }

  // Inflation classification (independent of corner validity).
  auto polygonArea = [&]() { return std::abs(signedArea(pts)); };
  const double area_meas = cluster.area > 0.0 ? cluster.area : polygonArea();
  const double area_pred =
    prediction.area > 0.0
      ? prediction.area
      : std::max(0.0, prediction.shape.dimensions.x * prediction.shape.dimensions.y);

  if (best_corner < 0) {
    attemptEndFace();  // no two-face corner -> try the single end-face anchor before giving up
    return g;
  }

  const size_t jc = static_cast<size_t>(best_corner);
  g.near_corner.x = pts[jc].x;
  g.near_corner.y = pts[jc].y;
  g.near_corner.z = cluster.pose.position.z;
  g.has_corner = true;

  // Grow the two arms along the contiguous ego-facing run, split at the corner.
  std::vector<Vec2> arm_a{pts[jc]};  // backward (along edges j-1, j-2, ...)
  for (size_t step = 0, i = (jc + n - 1) % n; step < n - 1 && facing[i];
       ++step, i = (i + n - 1) % n) {
    arm_a.push_back(pts[i]);
  }
  std::vector<Vec2> arm_b{pts[jc]};  // forward (along edges j, j+1, ...)
  for (size_t step = 0, i = jc; step < n - 1 && facing[i]; ++step, i = (i + 1) % n) {
    arm_b.push_back(pts[(i + 1) % n]);
  }

  const LineFit fa = fitLine(arm_a);
  const LineFit fb = fitLine(arm_b);
  if (!fa.valid || !fb.valid || fa.length < MIN_ARM_LEN || fb.length < MIN_ARM_LEN) {
    // Only one real face: the corner is not a genuine two-face junction. Drop it and reconstruct
    // the visible end face directly, so a thin rear/front cluster anchors the box at that face
    // instead of degrading to a centroid-as-center weak update.
    g.has_corner = false;
    g.trust = 0.0;
    attemptEndFace();
    return g;
  }

  const LineFit & longf = (fa.length >= fb.length) ? fa : fb;
  const LineFit & shortf = (fa.length >= fb.length) ? fb : fa;

  // Orientation cue from the long edge tangent, resolved against the predicted body axis.
  double long_dir = longf.dir;
  if (std::abs(normAngle(long_dir - pred_yaw)) > M_PI_2) long_dir = normAngle(long_dir + M_PI);
  const double straightness = std::exp(-(longf.residual / R_REF) * (longf.residual / R_REF));
  g.long_edge_dir = long_dir;
  g.long_edge_len = longf.length;
  g.roundness = std::clamp(1.0 - straightness, 0.0, 1.0);
  g.yaw_variance = BASE_YAW_VAR * (REF_LEN / std::max(longf.length, 1e-3)) *
                   (REF_LEN / std::max(longf.length, 1e-3)) / std::max(straightness, S_FLOOR);
  g.yaw_cue_valid = longf.length >= LONG_EDGE_MIN && straightness >= STRAIGHT_MIN &&
                    std::abs(normAngle(long_dir - pred_yaw)) <= ANGLE_MAX_TO_PRED;

  // Observed width: lateral extent of the whole visible hull, perpendicular to the long edge.
  const auto ext = computeOrientedExtent(pts, std::cos(long_dir), std::sin(long_dir));
  g.observed_width = ext.max_lat - ext.min_lat;
  g.observed_width_confidence =
    std::clamp(std::min(g.observed_width, pred_width) / pred_width, 0.0, 1.0) *
    std::clamp(shortf.length / pred_width, 0.0, 1.0);

  // Inflation classification.
  const double area_ratio = (area_meas > 0.0 && area_pred > 0.0)
                              ? std::max(area_meas, area_pred) / std::min(area_meas, area_pred)
                              : 1.0;
  if (area_ratio <= AREA_RATIO_INFLATE) {
    g.inflation = PolygonInflation::NONE;
  } else if (g.observed_width > pred_width * WIDTH_INFLATE_RATIO) {
    g.inflation = PolygonInflation::FAULTY;
  } else if (max_vertex_turn > SPIKE_TURN) {
    g.inflation = PolygonInflation::NOISE;
  } else {
    g.inflation = PolygonInflation::SHAPE_CHANGE_CANDIDATE;
  }

  // Overall trust: corner validity, modest straightness weighting, inflation penalty.
  double inflation_penalty = 1.0;
  switch (g.inflation) {
    case PolygonInflation::FAULTY:
      inflation_penalty = 0.5;
      break;
    case PolygonInflation::NOISE:
      inflation_penalty = 0.6;
      break;
    case PolygonInflation::SHAPE_CHANGE_CANDIDATE:
      inflation_penalty = 0.8;
      break;
    case PolygonInflation::NONE:
      inflation_penalty = 1.0;
      break;
  }
  g.trust = std::clamp((0.5 + 0.5 * straightness) * inflation_penalty, 0.0, 1.0);

  return g;
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
