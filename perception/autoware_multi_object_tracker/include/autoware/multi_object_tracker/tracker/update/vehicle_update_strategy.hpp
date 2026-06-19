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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__UPDATE__VEHICLE_UPDATE_STRATEGY_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__UPDATE__VEHICLE_UPDATE_STRATEGY_HPP_

#include "autoware/multi_object_tracker/object_model/shapes.hpp"
#include "autoware/multi_object_tracker/types.hpp"

namespace autoware::multi_object_tracker
{

// Discrete front/rear and left/right association of an observed near corner to the predicted body.
// Drives BicycleMotionModel::updateStatePoseCorner: `is_front` selects the front/rear endpoint
// blend, `s_lat` (+1 / -1) the lateral half-width sign. This is the ONLY place the prior pose
// enters the corner update — as a discrete choice, never as a mean injection (see
// PolygonMeasurement).
struct CornerAssociation
{
  bool is_front = false;
  double s_lat = 1.0;  // +1 / -1: which side of the body axis the corner sits on
};

// Associate an observed near corner (map frame) to the nearest predicted bounding-box corner. The
// box corner is picked purely by the sign of the corner's longitudinal / lateral offset from the
// predicted center, so the result is the front/rear + left/right quadrant the corner falls in.
CornerAssociation associateCornerToPrediction(
  const geometry_msgs::msg::Point & corner, const types::DynamicObject & prediction);

// Blends measurement position/orientation into pred using a distance-weighted scheme.
// When enlarge_covariance=true, inflates pose/velocity covariances for the weak-update path.
void createPseudoMeasurement(
  const types::DynamicObject & meas, types::DynamicObject & pred,
  const autoware_perception_msgs::msg::Shape & tracker_shape,
  const bool enlarge_covariance = false);

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__TRACKER__UPDATE__VEHICLE_UPDATE_STRATEGY_HPP_
