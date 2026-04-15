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

#include "autoware/multi_object_tracker/association/polar_association.hpp"

#include "autoware/multi_object_tracker/association/scoring/polar_scoring.hpp"
#include "autoware/multi_object_tracker/association/solver/gnn_solver.hpp"
#include "autoware/multi_object_tracker/types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <tf2/utils.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <list>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
namespace
{
constexpr double INVALID_SCORE = 0.0;

// Scoring weights: azimuth IoU is prioritized over radial and height
constexpr double W_AZIMUTH = 0.7;
constexpr double W_RADIAL = 0.2;
constexpr double W_HEIGHT = 0.1;

// Shape change detection thresholds (same semantics as BEV)
constexpr double AZIMUTH_IOU_SHAPE_CHECK_THRESHOLD = 0.7;
constexpr double AREA_RATIO_THRESHOLD = 1.3;

// Near-face gate: maximum allowed gap between the near faces of measurement and tracker [m].
// Enforces the LiDAR physics constraint that a cluster must originate from the closest visible
// surface of an object, not from its interior or far side.
constexpr double NEAR_FACE_GAP_THRESHOLD = 2.0;

// Azimuth bin helpers
// 24 bins × 15° (π/12 rad) each, covering the full [0, 2π) circle.
constexpr double kAzimuthBinWidth =
  2.0 * M_PI / static_cast<double>(PolarAssociation::kNumAzimuthBins);

/// Map any angle to a bin index in [0, kNumAzimuthBins).
int azimuthToBin(double angle)
{
  // Shift from [-π, π) to [0, 2π) then divide by bin width.
  double a = std::fmod(angle + M_PI, 2.0 * M_PI);
  if (a < 0.0) a += 2.0 * M_PI;
  const int bin = static_cast<int>(a / kAzimuthBinWidth);
  return std::min(bin, PolarAssociation::kNumAzimuthBins - 1);
}

/// Register tracker_idx to every bin that the azimuth interval covers.
void registerToAzimuthBins(
  std::array<std::vector<size_t>, PolarAssociation::kNumAzimuthBins> & bins,
  const polar_scoring::AzimuthInterval & azimuth, const size_t tracker_idx)
{
  // Number of bins the interval spans (always at least 1, capped at all bins).
  const int n = std::min(
    static_cast<int>(std::ceil(2.0 * azimuth.half_span / kAzimuthBinWidth)) + 1,
    PolarAssociation::kNumAzimuthBins);
  const int start_bin = azimuthToBin(azimuth.center - azimuth.half_span);
  for (int i = 0; i < n; ++i) {
    bins[(start_bin + i) % PolarAssociation::kNumAzimuthBins].push_back(tracker_idx);
  }
}

/// Collect unique tracker indices from all bins covered by the azimuth interval.
std::vector<size_t> queryAzimuthBins(
  const std::array<std::vector<size_t>, PolarAssociation::kNumAzimuthBins> & bins,
  const polar_scoring::AzimuthInterval & azimuth)
{
  const int n = std::min(
    static_cast<int>(std::ceil(2.0 * azimuth.half_span / kAzimuthBinWidth)) + 1,
    PolarAssociation::kNumAzimuthBins);
  const int start_bin = azimuthToBin(azimuth.center - azimuth.half_span);

  std::vector<size_t> candidates;
  for (int i = 0; i < n; ++i) {
    const auto & bin_vec = bins[(start_bin + i) % PolarAssociation::kNumAzimuthBins];
    candidates.insert(candidates.end(), bin_vec.begin(), bin_vec.end());
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  return candidates;
}

}  // namespace

using autoware_utils_debug::ScopedTimeTrack;

// -- Construction & configuration --

PolarAssociation::PolarAssociation(const AssociatorConfig & config)
: config_(config), score_threshold_(0.01)
{
  gnn_solver_ptr_ = std::make_unique<gnn_solver::MuSSP>();
}

void PolarAssociation::setEgoPose(const std::optional<geometry_msgs::msg::Pose> & ego_pose)
{
  ego_pose_ = ego_pose;
}

void PolarAssociation::setTimeKeeper(
  std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper_ptr)
{
  time_keeper_ = std::move(time_keeper_ptr);
}

// -- Top-level associate --

types::AssociationResult PolarAssociation::associate(
  const types::DynamicObjectList & measurements,
  const std::list<std::shared_ptr<Tracker>> & trackers)
{
  const types::AssociationData data = calcAssociationData(measurements, trackers);
  types::AssociationResult result;
  assign(data, result);
  return result;
}

// -- Data preparation --

PolarAssociation::PolarPreparationData PolarAssociation::prepareAssociationData(
  const types::DynamicObjectList & measurements,
  const std::list<std::shared_ptr<Tracker>> & trackers, const double ego_x, const double ego_y,
  const double ego_yaw)
{
  PolarPreparationData prep_data;
  const size_t num_trackers = trackers.size();

  prep_data.tracked_objects.reserve(num_trackers);
  prep_data.tracker_labels.reserve(num_trackers);
  prep_data.tracker_types.reserve(num_trackers);
  prep_data.tracker_footprints.reserve(num_trackers);

  // Extract tracked objects and build azimuth bin index
  {
    size_t tracker_idx = 0;
    for (auto & bin : azimuth_bins_) bin.clear();

    for (const auto & tracker : trackers) {
      types::DynamicObject tracked_object;
      tracker->getTrackedObject(measurements.header.stamp, tracked_object);

      // Compute polar footprint and register to all azimuth bins it covers
      prep_data.tracker_footprints.emplace_back(
        polar_scoring::computePolarFootprint(tracked_object, ego_x, ego_y, ego_yaw));
      registerToAzimuthBins(
        azimuth_bins_, prep_data.tracker_footprints.back().azimuth, tracker_idx);

      prep_data.tracked_objects.emplace_back(std::move(tracked_object));
      prep_data.tracker_labels.emplace_back(tracker->getHighestProbLabel());
      prep_data.tracker_types.emplace_back(tracker->getTrackerType());
      ++tracker_idx;
    }
  }

  return prep_data;
}

// -- Per-measurement scoring --

void PolarAssociation::processMeasurement(
  const types::DynamicObject & measurement_object, const size_t measurement_idx,
  const classes::Label measurement_label, const PolarPreparationData & prep_data,
  const double ego_x, const double ego_y, const double ego_yaw,
  types::AssociationData & association_data)
{
  const auto tracker_params_map_opt =
    get_map_value_if_exists(config_.association_params_map, measurement_label);
  if (!tracker_params_map_opt) return;
  const auto & tracker_params_map = tracker_params_map_opt->get();

  // Compute measurement polar footprint and query azimuth bins for candidate trackers
  const auto meas_fp =
    polar_scoring::computePolarFootprint(measurement_object, ego_x, ego_y, ego_yaw);
  const auto candidate_indices = queryAzimuthBins(azimuth_bins_, meas_fp.azimuth);

  for (const size_t tracker_idx : candidate_indices) {
    const auto tracker_type = prep_data.tracker_types[tracker_idx];

    const auto association_params_opt = get_map_value_if_exists(tracker_params_map, tracker_type);
    if (!association_params_opt) continue;
    const auto & association_params = association_params_opt->get();

    const auto & tracked_object = prep_data.tracked_objects[tracker_idx];
    const auto & tracker_fp = prep_data.tracker_footprints[tracker_idx];

    // Gate 1: Near-face alignment – the cluster's closest point must be near the tracker's
    // closest surface. LiDAR can only detect the nearest visible surface of a solid object,
    // so a cluster floating inside or at the far end of a tracker's bounding box is invalid.
    const double near_face_gap = std::abs(meas_fp.r_min - tracker_fp.r_min);
    if (near_face_gap > NEAR_FACE_GAP_THRESHOLD) continue;

    // Gate 2: Azimuth IoU (primary matching criterion)
    const double az_iou = polar_scoring::azimuthIoU(meas_fp.azimuth, tracker_fp.azimuth);

    // Gate 3: Radial compatibility
    const double rad_compat = polar_scoring::radialCompatibility(meas_fp.r_min, tracker_fp.r_min);

    // Gate 4: Height compatibility
    const double h_iou =
      polar_scoring::heightIoU(meas_fp.z_min, meas_fp.z_max, tracker_fp.z_min, tracker_fp.z_max);

    // Combined score with azimuth prioritized
    double raw_score = az_iou * (W_AZIMUTH + W_RADIAL * rad_compat + W_HEIGHT * h_iou);

    const double min_iou = association_params.min_iou;
    if (raw_score < min_iou) continue;

    // Normalize score to [0, 1], same convention as BEV
    const double score = (raw_score - min_iou) / (1.0 - min_iou);

    // Shape change detection for vehicle trackers
    bool has_significant_shape_change = false;
    if (az_iou < AZIMUTH_IOU_SHAPE_CHECK_THRESHOLD && isVehicleTrackerType(tracker_type)) {
      const double area_meas = measurement_object.area;
      const double area_trk = tracked_object.area;
      if (area_meas > 0.0 && area_trk > 0.0) {
        const double area_ratio = std::max(area_trk, area_meas) / std::min(area_trk, area_meas);
        if (area_ratio > AREA_RATIO_THRESHOLD) {
          has_significant_shape_change = true;
        }
      }
    }

    if (score > INVALID_SCORE) {
      association_data.entries.emplace_back(
        types::AssociationEntry{tracker_idx, measurement_idx, score, has_significant_shape_change});
    }
  }
}

// -- Full association data computation --

types::AssociationData PolarAssociation::calcAssociationData(
  const types::DynamicObjectList & measurements,
  const std::list<std::shared_ptr<Tracker>> & trackers)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  if (measurements.objects.empty() || trackers.empty() || !ego_pose_) {
    if (!ego_pose_) {
      static rclcpp::Clock steady_clock{RCL_STEADY_TIME};
      RCLCPP_WARN_THROTTLE(
        rclcpp::get_logger("polar_association"), steady_clock, 5000,
        "PolarAssociation: ego_pose not set, returning empty association");
    }
    return types::AssociationData{};
  }

  const double ego_x = ego_pose_->position.x;
  const double ego_y = ego_pose_->position.y;
  const double ego_yaw = tf2::getYaw(ego_pose_->orientation);

  auto prep_data = prepareAssociationData(measurements, trackers, ego_x, ego_y, ego_yaw);

  types::AssociationData association_data;
  association_data.tracker_uuids.reserve(trackers.size());
  association_data.measurement_uuids.reserve(measurements.objects.size());

  for (const auto & object : measurements.objects) {
    association_data.measurement_uuids.emplace_back(object.uuid);
  }
  for (const auto & tracker : trackers) {
    association_data.tracker_uuids.emplace_back(tracker->getUUID());
  }

  for (auto it = measurements.objects.begin(); it != measurements.objects.end(); ++it) {
    const size_t measurement_idx = std::distance(measurements.objects.begin(), it);
    const auto measurement_label = classes::getHighestProbLabel(it->classification);

    processMeasurement(
      *it, measurement_idx, measurement_label, prep_data, ego_x, ego_y, ego_yaw, association_data);
  }

  return association_data;
}

// -- Assignment (GNN solver) --

void PolarAssociation::assign(
  const types::AssociationData & data, types::AssociationResult & association_result)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  std::unordered_map<int, int> direct_assignment;
  std::unordered_map<int, int> reverse_assignment;

  std::vector<std::vector<double>> score = formatScoreMatrix(data);

  // Solve the linear assignment problem
  gnn_solver_ptr_->maximizeLinearAssignment(score, &direct_assignment, &reverse_assignment);

  // Build a lookup map for entries with significant shape change
  std::unordered_map<int, std::unordered_map<int, const types::AssociationEntry *>> entry_map;
  for (const auto & entry : data.entries) {
    if (entry.has_significant_shape_change && entry.score >= score_threshold_) {
      entry_map[static_cast<int>(entry.tracker_idx)][static_cast<int>(entry.measurement_idx)] =
        &entry;
    }
  }

  association_result.unassigned_trackers.reserve(data.tracker_uuids.size());
  association_result.unassigned_measurements.reserve(data.measurement_uuids.size());

  for (const auto & [tracker_idx, measurement_idx] : direct_assignment) {
    if (score[tracker_idx][measurement_idx] >= score_threshold_) {
      association_result.add(
        data.tracker_uuids[tracker_idx], data.measurement_uuids[measurement_idx]);

      auto tracker_it = entry_map.find(tracker_idx);
      if (tracker_it != entry_map.end()) {
        auto measurement_it = tracker_it->second.find(measurement_idx);
        if (measurement_it != tracker_it->second.end()) {
          association_result.trackers_with_shape_change.insert(data.tracker_uuids[tracker_idx]);
        }
      }
    }
  }

  for (size_t i = 0; i < data.tracker_uuids.size(); ++i) {
    auto it = direct_assignment.find(static_cast<int>(i));
    if (
      it == direct_assignment.end() || score[static_cast<int>(i)][it->second] < score_threshold_) {
      association_result.unassigned_trackers.emplace_back(data.tracker_uuids[i]);
    }
  }

  for (size_t i = 0; i < data.measurement_uuids.size(); ++i) {
    auto it = reverse_assignment.find(static_cast<int>(i));
    if (
      it == reverse_assignment.end() || score[it->second][static_cast<int>(i)] < score_threshold_) {
      association_result.unassigned_measurements.emplace_back(data.measurement_uuids[i]);
    }
  }
}

std::vector<std::vector<double>> PolarAssociation::formatScoreMatrix(
  const types::AssociationData & data) const
{
  std::vector<std::vector<double>> score_matrix(
    data.tracker_uuids.size(), std::vector<double>(data.measurement_uuids.size(), 0.0));
  for (const auto & entry : data.entries) {
    score_matrix[entry.tracker_idx][entry.measurement_idx] = entry.score;
  }
  return score_matrix;
}

}  // namespace autoware::multi_object_tracker
