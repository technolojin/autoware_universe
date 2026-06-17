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

#include "autoware/multi_object_tracker/association/association_manager.hpp"

#include <rclcpp/clock.hpp>
#include <rclcpp/logging.hpp>

#include <cmath>
#include <limits>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{

AssociationManager::AssociationManager(
  const TrackerAssociationConfig & association_config,
  const std::vector<types::InputChannel> & channels_config)
: channels_config_(channels_config),
  ego_pose_max_age_sec_(association_config.ego_pose_max_age_sec),
  bev_association_(std::make_unique<BevAssociation>(association_config)),
  polar_association_(std::make_unique<PolarAssociation>(association_config))
{
}

AssociationBase & AssociationManager::getAssociationForChannel(
  const uint channel_index, const bool polar_viable) const
{
  if (
    polar_viable &&
    channels_config_[channel_index].associator_type == types::AssociationType::POLAR) {
    return *polar_association_;
  }
  return *bev_association_;
}

bool AssociationManager::isPolarViable(
  const std::optional<geometry_msgs::msg::PoseStamped> & ego_pose,
  const rclcpp::Time & measurement_time) const
{
  if (!ego_pose.has_value()) return false;
  const rclcpp::Time ego_time{ego_pose->header.stamp};
  const double dt = std::abs((measurement_time - ego_time).seconds());
  return dt <= ego_pose_max_age_sec_;
}

types::AssociationResult AssociationManager::associate(
  const types::DynamicObjectList & measurements,
  const std::list<std::shared_ptr<Tracker>> & trackers,
  const std::optional<geometry_msgs::msg::PoseStamped> & ego_pose)
{
  polar_association_->setEgoPose(ego_pose ? std::make_optional(ego_pose->pose) : std::nullopt);

  const rclcpp::Time meas_time{measurements.header.stamp};
  const bool polar_viable = isPolarViable(ego_pose, meas_time);

  const bool channel_wants_polar =
    channels_config_[measurements.channel_index].associator_type == types::AssociationType::POLAR;
  if (channel_wants_polar && !polar_viable) {
    const double dt = ego_pose ? (meas_time - rclcpp::Time{ego_pose->header.stamp}).seconds()
                               : std::numeric_limits<double>::infinity();
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("association_manager"), steady_clock_, 5000,
      "AssociationManager: polar channel falling back to BEV — ego pose dt=%.3f s (threshold=%.3f "
      "s)",
      dt, ego_pose_max_age_sec_);
  }

  AssociationBase & geometric_associator =
    getAssociationForChannel(measurements.channel_index, polar_viable);

  // DetectedObjects channels carry no stable upstream id: associate purely geometrically as before.
  const auto & channel = channels_config_[measurements.channel_index];
  if (channel.type != types::InputMessageType::TRACKED_OBJECTS) {
    return geometric_associator.associate(measurements, trackers);
  }

  // TrackedObjects channel: resolve identity by upstream UUID first, geometry fills the rest.
  return associateTrackedByUuid(measurements, trackers, geometric_associator, meas_time);
}

types::AssociationResult AssociationManager::associateTrackedByUuid(
  const types::DynamicObjectList & measurements,
  const std::list<std::shared_ptr<Tracker>> & trackers, AssociationBase & geometric_associator,
  const rclcpp::Time & measurement_time) const
{
  const uint channel_index = measurements.channel_index;

  // Build the transient per-channel lookup from the snapshot's (non-stale) bindings. Rebuilt every
  // call on purpose: no shared mutable index -> lock-free across parallel associators.
  std::unordered_map<
    unique_identifier_msgs::msg::UUID, Tracker *, types::UUIDHash, types::UUIDEqual>
    uuid_to_tracker;
  for (const auto & tracker : trackers) {
    for (const auto & binding : tracker->getSourceBindings()) {
      if (binding.channel == channel_index && !binding.isStale(measurement_time)) {
        uuid_to_tracker[binding.uuid] = tracker.get();
      }
    }
  }

  types::AssociationResult result;

  // Resolve UUID matches. A tracker can be claimed by at most one measurement per batch.
  std::unordered_set<const Tracker *> matched_trackers;
  std::vector<bool> measurement_matched(measurements.objects.size(), false);
  if (!uuid_to_tracker.empty()) {
    for (size_t i = 0; i < measurements.objects.size(); ++i) {
      const auto & measurement = measurements.objects[i];
      if (!measurement.has_source_uuid) continue;
      const auto it = uuid_to_tracker.find(measurement.source_uuid);
      if (it == uuid_to_tracker.end()) continue;
      const Tracker * tracker = it->second;
      if (matched_trackers.count(tracker)) continue;
      result.add(tracker->getUUID(), measurement.uuid);
      matched_trackers.insert(tracker);
      measurement_matched[i] = true;
    }
  }

  // Build residuals (everything not UUID-matched) for the geometric fallback.
  types::DynamicObjectList residual_measurements;
  residual_measurements.header = measurements.header;
  residual_measurements.channel_index = measurements.channel_index;
  for (size_t i = 0; i < measurements.objects.size(); ++i) {
    if (!measurement_matched[i]) {
      residual_measurements.objects.push_back(measurements.objects[i]);
    }
  }
  residual_measurements.buildUuidIndex();

  std::list<std::shared_ptr<Tracker>> residual_trackers;
  for (const auto & tracker : trackers) {
    if (!matched_trackers.count(tracker.get())) {
      residual_trackers.push_back(tracker);
    }
  }

  // Fast exit: nothing left for geometry (steady state — every measurement matched a binding).
  if (residual_measurements.objects.empty()) {
    result.unassigned_trackers.reserve(residual_trackers.size());
    for (const auto & tracker : residual_trackers) {
      result.unassigned_trackers.push_back(tracker->getUUID());
    }
    return result;
  }

  const types::AssociationResult geometric_result =
    geometric_associator.associate(residual_measurements, residual_trackers);

  // Merge geometric pairs and metadata into the UUID-resolved result. Residuals exclude the
  // UUID-matched trackers/measurements, so the assigned/unassigned sets compose without overlap.
  for (const auto & [tracker_uuid, measurement_uuid] : geometric_result.tracker_to_measurement) {
    result.add(tracker_uuid, measurement_uuid);
  }
  for (const auto & tracker_uuid : geometric_result.trackers_with_shape_change) {
    result.trackers_with_shape_change.insert(tracker_uuid);
  }
  result.unassigned_trackers = geometric_result.unassigned_trackers;
  result.unassigned_measurements = geometric_result.unassigned_measurements;

  return result;
}

void AssociationManager::setTimeKeeper(
  std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_ptr)
{
  polar_association_->setTimeKeeper(time_keeper_ptr);
  bev_association_->setTimeKeeper(std::move(time_keeper_ptr));
}

}  // namespace autoware::multi_object_tracker
