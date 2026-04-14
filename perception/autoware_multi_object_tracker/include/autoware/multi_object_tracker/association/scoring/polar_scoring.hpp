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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__ASSOCIATION__SCORING__POLAR_SCORING_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__ASSOCIATION__SCORING__POLAR_SCORING_HPP_

#include "autoware/multi_object_tracker/types.hpp"

#include <cmath>
#include <vector>

namespace autoware::multi_object_tracker
{
namespace polar_scoring
{

/// Azimuth interval in center + half-span form.
/// This representation handles angle wrapping naturally: the angular distance
/// between two centers is always computed via normalizeAngle(), so intervals
/// near the +/-pi boundary compare correctly without explicit wrapping logic.
struct AzimuthInterval
{
  double center;     // center angle [rad], in [-pi, pi]
  double half_span;  // half angular width [rad], >= 0
};

/// Polar footprint of an object as seen from the ego vehicle.
struct PolarFootprint
{
  AzimuthInterval azimuth;
  double r_min;  // minimum range from ego [m]
  double r_max;  // maximum range from ego [m]
  double z_min;  // minimum height [m]
  double z_max;  // maximum height [m]
};

/// Normalize angle to [-pi, pi].
inline double normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

/// Compute polar footprint of an object relative to ego position and heading.
/// Object pose and shape are in the map frame; the result is in ego-centric polar coordinates.
/// @param object  Dynamic object with pose and shape in map frame
/// @param ego_x   Ego position x in map frame [m]
/// @param ego_y   Ego position y in map frame [m]
/// @param ego_yaw Ego heading in map frame [rad]
PolarFootprint computePolarFootprint(
  const types::DynamicObject & object, double ego_x, double ego_y, double ego_yaw);

/// 1D IoU of two azimuth intervals.
/// Uses the center+half_span representation to handle angle wrapping naturally.
/// @return value in [0, 1]
double azimuthIoU(const AzimuthInterval & a, const AzimuthInterval & b);

/// Radial front-point compatibility.
/// @return value in [0, 1]
double radialCompatibility(double r_min_a, double r_min_b);

/// Height (z-axis) IoU.
/// @return value in [0, 1]
double heightIoU(double z_min_a, double z_max_a, double z_min_b, double z_max_b);

/// Fraction of the target interval that is NOT occluded by the occluder.
/// @return value in [0, 1]: 1.0 = fully visible, 0.0 = fully occluded
double visibleFraction(const AzimuthInterval & target, const AzimuthInterval & occluder);

}  // namespace polar_scoring
}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__ASSOCIATION__SCORING__POLAR_SCORING_HPP_
