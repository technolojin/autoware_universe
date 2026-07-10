// Copyright 2020 TIER IV, Inc.
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

#ifndef AUTOWARE__MULTI_OBJECT_TRACKER__CONFIGURATIONS_HPP_
#define AUTOWARE__MULTI_OBJECT_TRACKER__CONFIGURATIONS_HPP_

#include "autoware/multi_object_tracker/types.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace autoware::multi_object_tracker
{

struct EnumClassHash
{
  template <typename T>
  std::size_t operator()(const T value) const
  {
    return static_cast<std::size_t>(value);
  }
};

//// Association profile
struct AssociationProfile
{
  double max_dist_sq;
  double max_area;
  double min_area;
  double min_iou;
};

using LabelDoubleMap = std::unordered_map<classes::Label, double, EnumClassHash>;
using LabelBoolMap = std::unordered_map<classes::Label, bool, EnumClassHash>;

using ShapeLabelKey = std::pair<types::ShapeType, classes::Label>;
struct ShapeLabelKeyHash
{
  std::size_t operator()(const ShapeLabelKey & k) const
  {
    const auto h1 = std::hash<uint8_t>{}(static_cast<uint8_t>(k.first));
    const auto h2 = std::hash<uint8_t>{}(static_cast<uint8_t>(k.second));
    return h1 ^ (h2 << 8);
  }
};

using AssociationProfileMap =
  std::unordered_map<types::TrackerType, AssociationProfile, EnumClassHash>;

using AssociationMap = std::unordered_map<ShapeLabelKey, AssociationProfileMap, ShapeLabelKeyHash>;

using ShapeLabelToTrackerTypeMap =
  std::unordered_map<ShapeLabelKey, types::TrackerType, ShapeLabelKeyHash>;

//// Per-tracker-type configuration (parallels tracker_profiles: keyed by tracker type)
struct PolygonTrackerConfig
{
  bool enable_velocity_estimation{true};
  // Whether to publish estimated velocity, per object classification.
  // A label absent from the map (or set false) means motion output is suppressed for that label.
  LabelBoolMap enable_motion_output;
};

struct StaticTrackerConfig
{
  bool convert_polygon_to_bbox{false};
};

//// Localization (ego-pose) error model.
//// A plain TF lookup carries no covariance, so the ego pose it yields is treated as exact. This
//// model supplies the ego-pose uncertainty for the TF ego-source path (and the odometry-source
//// buffer-miss fallback), which addOdometryUncertainty() folds into each detection's measurement
//// covariance R. The heading term is what stops the strong bicycle motion model from turning an
//// unmodeled ego-heading error into a persistent wrong object yaw. Values are 1-sigma stddevs.
struct LocalizationErrorModel
{
  double pos_stddev_x{0.3};  // [m] ego position stddev along map x
  double pos_stddev_y{0.3};  // [m] ego position stddev along map y
  double yaw_stddev{0.017};  // [rad] ego heading stddev (~1 deg)
  // Correlation time [s] over which the ego pose covariance is leaked into the tracked-object
  // position process noise during prediction (per-step contribution scales with dt / correlation
  // time). Larger => slower injection. Non-positive disables the prediction-side pose term.
  double correlation_time{1.0};
};

struct TrackerConfigs
{
  PolygonTrackerConfig polygon_tracker;
  StaticTrackerConfig static_tracker;
};

//// Tracker creation (spawning, type mapping)
struct TrackerCreationConfig
{
  ShapeLabelToTrackerTypeMap shape_tracker_map;
  std::unordered_set<ShapeLabelKey, ShapeLabelKeyHash> explicit_null_combos;

  void setCreation(types::ShapeType shape, classes::Label label, types::TrackerType tracker_type)
  {
    shape_tracker_map[{shape, label}] = tracker_type;
  }

  void setExplicitNull(types::ShapeType shape, classes::Label label)
  {
    explicit_null_combos.insert({shape, label});
  }
};

//// Tracker association (measurement <-> tracker matching)
struct TrackerAssociationConfig
{
  double unknown_association_giou_threshold{0.0};
  double score_threshold{0.01};
  double ego_pose_max_age_sec{0.21};

  AssociationMap association_params_map;
  LabelDoubleMap max_dist_sq_per_label;

  void setProfile(
    types::ShapeType shape, classes::Label label, types::TrackerType tracker_type,
    AssociationProfile profile)
  {
    association_params_map[{shape, label}][tracker_type] = profile;
  }

  void buildMaxDistances()
  {
    max_dist_sq_per_label.clear();
    for (const auto & [shape_label, profile_map] : association_params_map) {
      auto & max_sq = max_dist_sq_per_label[shape_label.second];
      for (const auto & [tracker_type, profile] : profile_map) {
        (void)tracker_type;
        max_sq = std::max(max_sq, profile.max_dist_sq);
      }
    }
  }
};

//// Tracker overlap manager (tracker-to-tracker layer: remove spatially redundant trackers)
struct TrackerOverlapManagerConfig
{
  float min_known_object_removal_iou{0.0f};
  float min_unknown_object_removal_iou{0.0f};
  double pruning_giou_threshold{0.0};
  LabelDoubleMap pruning_distance_thresholds;
  LabelDoubleMap pruning_distance_thresholds_sq;
};

//// Utility: safe map lookup
template <typename Map, typename Key>
auto get_map_value_if_exists(const Map & map, const Key & key)
  -> std::optional<std::reference_wrapper<const typename Map::mapped_type>>
{
  const auto it = map.find(key);
  if (it == map.end()) {
    return std::nullopt;
  }
  return std::cref(it->second);
}

}  // namespace autoware::multi_object_tracker

#endif  // AUTOWARE__MULTI_OBJECT_TRACKER__CONFIGURATIONS_HPP_
