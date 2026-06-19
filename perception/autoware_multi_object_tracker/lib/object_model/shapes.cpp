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
// --- shared helpers for the polygon analyzers -------------------------------------------------

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

}  // namespace

namespace
{
// --- helpers for analyzePolygonMeasurement -------------------------------------------------------
//
// The whole analyzer is parameter-light by design: the geometry (corner position, yaw, extents) is
// a pure function of the measured points, and only the COVARIANCE magnitude depends on a single
// sensor constant below. There are no scene-tuned thresholds — the distinct-faces decision is a
// statistical significance test, and bad measurements are left for the EKF innovation gate.

constexpr double LIDAR_POINT_STD = 0.10;  // [m] lateral scatter of a cluster point about the true
                                          // surface (sensor range noise + clustering jitter). The
                                          // ONE physical knob: it scales the measurement covariance
                                          // and sets the flatness floor below; corner/yaw means are
                                          // independent of it.
constexpr double FACE_K_SIGMA = 3.0;  // two faces are "distinct" when their directions differ by
                                      // more than this many combined sigma (universal, not tuned)
constexpr double FACE_MAX_CURVATURE =
  0.06;  // [-] max RMS-residual / arm-length for a run to count as a flat face. Dimensionless and
         // scale-free: a straight surface (even with a rounded join) stays well below it while a
         // curved arc exceeds it, so a rounded BODY never poses as a two-face corner.

// Total-least-squares fit of a single face, carrying the uncertainty needed to propagate a corner
// covariance. dir_var is the variance of the fitted direction: rotating the far end of a span by
// dtheta moves a point at along-distance l perpendicular by l*dtheta, so the angular error std is
// (per-point perpendicular noise) / sqrt(Sum l^2).
struct FaceFit
{
  Vec2 centroid{};
  double dir = 0.0;         // principal direction [rad]
  double sigma_perp = 0.0;  // RMS perpendicular residual [m]
  double dir_var = 0.0;     // variance of dir [rad^2]
  double length = 0.0;      // extent along dir [m]
  size_t n = 0;
  bool valid = false;
};

FaceFit fitFace(const std::vector<Vec2> & pts)
{
  FaceFit f;
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
  double sq_perp = 0.0, sum_along2 = 0.0;
  double min_a = std::numeric_limits<double>::max(), max_a = std::numeric_limits<double>::lowest();
  for (const auto & p : pts) {
    const double dx = p.x - cx, dy = p.y - cy;
    const double along = dx * cd + dy * sd;
    const double perp = -dx * sd + dy * cd;
    sq_perp += perp * perp;
    sum_along2 += along * along;
    min_a = std::min(min_a, along);
    max_a = std::max(max_a, along);
  }
  f.centroid = {cx, cy};
  f.n = n;
  f.length = max_a - min_a;
  f.sigma_perp = std::sqrt(sq_perp / static_cast<double>(n));
  // Floor the per-point perpendicular noise at the sensor std so a perfectly collinear arm (e.g.
  // n = 2) still reports a finite, honest direction uncertainty rather than zero.
  const double s_perp2 =
    std::max(sq_perp / static_cast<double>(n), LIDAR_POINT_STD * LIDAR_POINT_STD);
  f.dir_var = s_perp2 / std::max(sum_along2, 1e-6);
  f.valid = true;
  return f;
}

// Intersection of two lines given as (centroid, dir). Returns false when near-parallel.
bool lineIntersect(const FaceFit & a, const FaceFit & b, Vec2 & out)
{
  const double c1 = std::cos(a.dir), s1 = std::sin(a.dir);
  const double c2 = std::cos(b.dir), s2 = std::sin(b.dir);
  const double denom = c1 * s2 - s1 * c2;  // cross product of the two direction unit vectors
  if (std::abs(denom) < 1e-6) return false;
  const double dx = b.centroid.x - a.centroid.x;
  const double dy = b.centroid.y - a.centroid.y;
  const double t = (dx * s2 - dy * c2) / denom;
  out.x = a.centroid.x + t * c1;
  out.y = a.centroid.y + t * s1;
  return true;
}

// Corner covariance from the two face fits. Each face constrains the corner perpendicular to its
// tangent with variance (line offset variance) + (lever along the face)^2 * (direction variance);
// the lever term is why a corner far from a face's sampled span is less certain. Summing the two
// rank-1 information contributions and inverting yields a 2x2 covariance that is well-conditioned
// for perpendicular faces and grows along the ill-determined axis as the faces become parallel.
std::array<double, 4> cornerCovariance(const FaceFit & a, const FaceFit & b, const Vec2 & corner)
{
  Eigen::Matrix2d info = Eigen::Matrix2d::Zero();
  for (const FaceFit * f : {&a, &b}) {
    const double cd = std::cos(f->dir), sd = std::sin(f->dir);
    const double lever = (corner.x - f->centroid.x) * cd + (corner.y - f->centroid.y) * sd;
    const double s_perp2 =
      std::max(f->sigma_perp * f->sigma_perp, LIDAR_POINT_STD * LIDAR_POINT_STD);
    const double offset_var = s_perp2 / static_cast<double>(f->n);
    const double perp_var = offset_var + lever * lever * f->dir_var;
    const Eigen::Vector2d nrm(-sd, cd);  // face normal
    info += (1.0 / std::max(perp_var, 1e-9)) * (nrm * nrm.transpose());
  }
  const Eigen::Matrix2d cov = info.inverse();
  return {cov(0, 0), cov(0, 1), cov(1, 0), cov(1, 1)};
}
}  // namespace

PolygonMeasurement analyzePolygonMeasurement(
  const types::DynamicObject & cluster, const geometry_msgs::msg::Point & ego_pos,
  const types::DynamicObject & prediction)
{
  PolygonMeasurement m;

  const auto & raw = cluster.shape.footprint.points;
  if (cluster.shape.type != autoware_perception_msgs::msg::Shape::POLYGON || raw.size() < 3) {
    return m;
  }

  std::vector<Vec2> pts = footprintToMap(cluster);
  const size_t n = pts.size();
  if (signedArea(pts) < 0.0) std::reverse(pts.begin(), pts.end());  // enforce CCW

  // Ego-facing edge mask: for a CCW polygon the outward normal of edge (e_x,e_y) is (e_y,-e_x); the
  // edge faces ego when it points toward ego.
  std::vector<bool> facing(n, false);
  for (size_t i = 0; i < n; ++i) {
    const auto & p0 = pts[i];
    const auto & p1 = pts[(i + 1) % n];
    const double ex = p1.x - p0.x, ey = p1.y - p0.y;
    const double mx = 0.5 * (p0.x + p1.x), my = 0.5 * (p0.y + p1.y);
    facing[i] = (ey * (ego_pos.x - mx) - ex * (ego_pos.y - my)) > 0.0;
  }

  // Longest contiguous (cyclic) run of ego-facing edges. A fully ego-facing hull is degenerate.
  bool all_facing = true;
  for (size_t i = 0; i < n; ++i) {
    if (!facing[i]) {
      all_facing = false;
      break;
    }
  }
  if (all_facing) return m;
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
  if (best_run < 1) return m;

  // Vertices spanned by the run (best_run edges -> best_run + 1 vertices).
  std::vector<Vec2> run;
  run.reserve(best_run + 1);
  for (size_t t = 0; t <= best_run; ++t) run.push_back(pts[(best_start + t) % n]);

  const double pred_yaw = tf2::getYaw(prediction.pose.orientation);
  auto resolveBranch = [&](double dir) {
    return (std::abs(normAngle(dir - pred_yaw)) > M_PI_2) ? normAngle(dir + M_PI) : dir;
  };

  // Split the run into two faces at the bend that minimizes the combined squared residual. `s` is
  // the shared corner vertex, so each side keeps >= 2 vertices (a sharp L is the minimal 3-vertex
  // run, s = 1). The best split is accepted as a corner only when the two face directions are
  // statistically distinct (a real bend, not noise on one straight face) and the lines intersect.
  FaceFit best_a, best_b;
  bool have_corner = false;
  if (run.size() >= 3) {
    double best_sse = std::numeric_limits<double>::max();
    for (size_t s = 1; s + 1 < run.size(); ++s) {
      const std::vector<Vec2> side_a(run.begin(), run.begin() + s + 1);  // [0, s], shares vertex s
      const std::vector<Vec2> side_b(run.begin() + s, run.end());        // [s, end]
      const FaceFit fa = fitFace(side_a);
      const FaceFit fb = fitFace(side_b);
      if (!fa.valid || !fb.valid) continue;
      const double sse = fa.sigma_perp * fa.sigma_perp * static_cast<double>(fa.n) +
                         fb.sigma_perp * fb.sigma_perp * static_cast<double>(fb.n);
      if (sse < best_sse) {
        best_sse = sse;
        best_a = fa;
        best_b = fb;
      }
    }
    if (best_a.valid && best_b.valid) {
      const double dtheta = std::abs(normAngle(best_a.dir - best_b.dir));
      const double sigma = std::sqrt(best_a.dir_var + best_b.dir_var);
      // A genuine corner needs (1) two statistically distinct directions AND (2) both faces flat.
      // The curvature gate is what separates a real bend from gentle curvature: an arc can be split
      // into two arbitrarily distinct half-directions, but each half keeps a systematic residual
      // proportional to its length (it bows), so it fails (2); a straight face does not.
      const bool distinct = dtheta > FACE_K_SIGMA * sigma;
      const bool flat = best_a.sigma_perp <= FACE_MAX_CURVATURE * best_a.length &&
                        best_b.sigma_perp <= FACE_MAX_CURVATURE * best_b.length;
      Vec2 corner;
      if (distinct && flat && lineIntersect(best_a, best_b, corner)) {
        m.has_corner = true;
        m.corner.x = corner.x;
        m.corner.y = corner.y;
        m.corner.z = cluster.pose.position.z;
        m.corner_cov = cornerCovariance(best_a, best_b, corner);
        have_corner = true;
      }
    }
  }

  // Heading + visible extents. With a corner, the longer arm is the body-longitudinal cue; without
  // one, a single straight run still yields a tangent (used as a weak heading hint only).
  const FaceFit yaw_fit =
    have_corner ? (best_a.length >= best_b.length ? best_a : best_b) : fitFace(run);
  if (yaw_fit.valid && yaw_fit.length > 0.0) {
    m.has_yaw = true;
    m.yaw = resolveBranch(yaw_fit.dir);
    m.yaw_var = yaw_fit.dir_var;
    const auto ext = computeOrientedExtent(run, std::cos(m.yaw), std::sin(m.yaw));
    m.visible_length = ext.max_along - ext.min_along;
    m.visible_width = ext.max_lat - ext.min_lat;
  }

  return m;
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
