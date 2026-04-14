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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__ASSOCIATION__POLAR_ASSOCIATION_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__ASSOCIATION__POLAR_ASSOCIATION_HPP_

#include "autoware/multi_object_tracker/association/association_base.hpp"
#include "autoware/multi_object_tracker/association/scoring/polar_scoring.hpp"
#include "autoware/multi_object_tracker/association/solver/gnn_solver.hpp"
#include "autoware/multi_object_tracker/configurations.hpp"
#include "autoware/multi_object_tracker/tracker/model/tracker_base.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <autoware_utils_debug/time_keeper.hpp>

#include <geometry_msgs/msg/pose.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{

// Spatial index types for R-tree tracker lookup (same as BevAssociation)
namespace polar_detail
{
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::point<double, 2, bg::cs::cartesian> Point;
typedef bg::model::box<Point> Box;
typedef std::pair<Point, size_t> ValueType;
}  // namespace polar_detail

/// Polar-coordinate association algorithm.
/// Converts measurement and tracker bounding boxes to polar coordinates centered at the ego
/// vehicle (base_link origin in map frame), then scores overlap based on azimuth interval IoU.
/// Handles the case where a thin lidar cluster (e.g., truck wall) has poor BEV IoU with a
/// tracked 3D bounding box but subtends the same angular extent from the ego's perspective.
/// Assigned per input channel via InputChannel::associator_type = AssociationType::POLAR.
class PolarAssociation : public AssociationBase
{
public:
  explicit PolarAssociation(const AssociatorConfig & config);
  ~PolarAssociation() override = default;

  /// AssociationBase implementation.
  types::AssociationResult associate(
    const types::DynamicObjectList & measurements,
    const std::list<std::shared_ptr<Tracker>> & trackers) override;

  /// Set ego pose for polar coordinate computation. Must be called before associate().
  void setEgoPose(const std::optional<geometry_msgs::msg::Pose> & ego_pose);

  void setTimeKeeper(std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_ptr);

private:
  // Per-tracker precomputed data for a single association round
  struct PolarPreparationData
  {
    std::vector<types::DynamicObject> tracked_objects;
    std::vector<classes::Label> tracker_labels;
    std::vector<types::TrackerType> tracker_types;
    std::vector<polar_scoring::PolarFootprint> tracker_footprints;
    std::vector<double> visibility_ratios;  // per-tracker [0,1] after occlusion
  };

  AssociatorConfig config_;
  const double score_threshold_;
  std::unique_ptr<gnn_solver::GnnSolverInterface> gnn_solver_ptr_;
  std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_;
  std::optional<geometry_msgs::msg::Pose> ego_pose_;

  // R-tree for spatial indexing of trackers
  boost::geometry::index::rtree<polar_detail::ValueType, boost::geometry::index::quadratic<16>>
    rtree_;
  // Maximum squared search distance per measurement class
  AssociatorConfig::LabelDoubleMap max_squared_dist_per_class_;

  void updateMaxSearchDistances();

  PolarPreparationData prepareAssociationData(
    const types::DynamicObjectList & measurements,
    const std::list<std::shared_ptr<Tracker>> & trackers, double ego_x, double ego_y,
    double ego_yaw);

  void processMeasurement(
    const types::DynamicObject & measurement_object, size_t measurement_idx,
    classes::Label measurement_label, const PolarPreparationData & prep_data, double ego_x,
    double ego_y, double ego_yaw, types::AssociationData & association_data);

  types::AssociationData calcAssociationData(
    const types::DynamicObjectList & measurements,
    const std::list<std::shared_ptr<Tracker>> & trackers);

  void assign(const types::AssociationData & data, types::AssociationResult & result);

  std::vector<std::vector<double>> formatScoreMatrix(const types::AssociationData & data) const;
};

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__ASSOCIATION__POLAR_ASSOCIATION_HPP_
