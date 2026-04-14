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

#include "autoware/multi_object_tracker/association/scoring/polar_scoring.hpp"

#include "autoware/multi_object_tracker/object_model/shapes.hpp"

#include <autoware_utils_geometry/boost_geometry.hpp>
#include <autoware_utils_geometry/boost_polygon_utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace autoware::multi_object_tracker
{
namespace polar_scoring
{

namespace
{
constexpr double MIN_RANGE = 1.0;  // Minimum range to avoid azimuth instability [m]
constexpr double MIN_SPAN = 1e-6;  // Minimum span to avoid division by zero
}  // namespace

PolarFootprint computePolarFootprint(
  const types::DynamicObject & object, const double ego_x, const double ego_y,
  const double ego_yaw)
{
  // Get the 2D polygon corners in map frame
  const auto polygon = autoware_utils_geometry::to_polygon2d(object.pose, object.shape);
  const auto & points = polygon.outer();

  // Height range from object shape
  const auto [z_min, z_max] = shapes::getObjectZRange(object);

  if (points.size() < 2) {
    // Degenerate polygon: use object center as a single point
    const double dx = object.pose.position.x - ego_x;
    const double dy = object.pose.position.y - ego_y;
    const double range = std::max(MIN_RANGE, std::sqrt(dx * dx + dy * dy));
    const double azimuth = normalizeAngle(std::atan2(dy, dx) - ego_yaw);
    return {{azimuth, 0.0}, range, range, z_min, z_max};
  }

  // Compute centroid-based azimuth as the reference center
  const double cx = object.pose.position.x - ego_x;
  const double cy = object.pose.position.y - ego_y;
  const double center_azimuth = normalizeAngle(std::atan2(cy, cx) - ego_yaw);

  double r_min = std::numeric_limits<double>::max();
  double r_max = 0.0;
  double max_offset = 0.0;

  for (const auto & pt : points) {
    const double dx = pt.x() - ego_x;
    const double dy = pt.y() - ego_y;
    const double range = std::sqrt(dx * dx + dy * dy);
    r_min = std::min(r_min, range);
    r_max = std::max(r_max, range);

    // Skip corners too close to ego for stable azimuth computation
    if (range < MIN_RANGE) continue;

    const double azimuth = normalizeAngle(std::atan2(dy, dx) - ego_yaw);
    const double offset = std::abs(normalizeAngle(azimuth - center_azimuth));
    max_offset = std::max(max_offset, offset);
  }

  r_min = std::max(r_min, MIN_RANGE);

  return {{center_azimuth, max_offset}, r_min, r_max, z_min, z_max};
}

double azimuthIoU(const AzimuthInterval & a, const AzimuthInterval & b)
{
  // Angular distance between centers, handling wrapping
  const double d = std::abs(normalizeAngle(a.center - b.center));

  // In local coordinates: A = [-h_A, h_A], B = [d - h_B, d + h_B]
  const double left = std::max(-a.half_span, d - b.half_span);
  const double right = std::min(a.half_span, d + b.half_span);
  const double intersection = std::max(0.0, right - left);
  const double union_span = 2.0 * a.half_span + 2.0 * b.half_span - intersection;

  if (union_span < MIN_SPAN) return 0.0;
  return std::min(1.0, intersection / union_span);
}

double radialCompatibility(
  const double r_min_a, const double r_max_a, const double r_min_b, const double r_max_b)
{
  const double overlap = std::max(0.0, std::min(r_max_a, r_max_b) - std::max(r_min_a, r_min_b));
  const double span = std::max(r_max_a, r_max_b) - std::min(r_min_a, r_min_b);
  if (span < MIN_SPAN) return 0.0;
  return std::min(1.0, overlap / span);
}

double heightIoU(
  const double z_min_a, const double z_max_a, const double z_min_b, const double z_max_b)
{
  const double overlap = std::max(0.0, std::min(z_max_a, z_max_b) - std::max(z_min_a, z_min_b));
  const double span = std::max(z_max_a, z_max_b) - std::min(z_min_a, z_min_b);
  if (span < MIN_SPAN) return 0.0;
  return std::min(1.0, overlap / span);
}

double visibleFraction(const AzimuthInterval & target, const AzimuthInterval & occluder)
{
  if (target.half_span < MIN_SPAN) return 0.0;

  const double d = std::abs(normalizeAngle(target.center - occluder.center));

  // Compute the overlap between target and occluder in local coords
  const double left = std::max(-target.half_span, d - occluder.half_span);
  const double right = std::min(target.half_span, d + occluder.half_span);
  const double occluded = std::max(0.0, right - left);

  return 1.0 - occluded / (2.0 * target.half_span);
}

}  // namespace polar_scoring
}  // namespace autoware::multi_object_tracker
