// Copyright 2021 TIER IV, Inc.
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

#include "autoware/map_based_prediction/map_based_prediction_node/callbacks.hpp"

#include "autoware/map_based_prediction/map_based_prediction_node/diagnostics.hpp"
#include "autoware/map_based_prediction/utils.hpp"

#include <autoware/lanelet2_utils/conversion.hpp>
#include <autoware/lanelet2_utils/geometry.hpp>
#include <autoware/object_recognition_utils/object_recognition_utils.hpp>
#include <autoware_utils/autoware_utils.hpp>
#include <autoware_utils/ros/uuid_helper.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <utility>

namespace autoware::map_based_prediction
{
using autoware_utils::ScopedTimeTrack;

// ---------------------------------------------------------------------------
// MapCallback
// ---------------------------------------------------------------------------

MapCallback::MapCallback(autoware::agnocast_wrapper::Node * node, NodeState & state)
: node_(node), state_(state)
{
}

void MapCallback::mapCallback(const AUTOWARE_MESSAGE_CONST_SHARED_PTR(LaneletMapBin) & msg)
{
  RCLCPP_DEBUG(node_->get_logger(), "[Map Based Prediction]: Start loading lanelet");

  state_.lanelet_map_ptr = autoware::experimental::lanelet2_utils::remove_const(
    autoware::experimental::lanelet2_utils::from_autoware_map_msgs(*msg));

  auto routing_graph_and_traffic_rules =
    autoware::experimental::lanelet2_utils::instantiate_routing_graph_and_traffic_rules(
      state_.lanelet_map_ptr);

  auto routing_graph_ptr =
    autoware::experimental::lanelet2_utils::remove_const(routing_graph_and_traffic_rules.first);
  auto traffic_rules_ptr = routing_graph_and_traffic_rules.second;

  state_.predictor_vehicle->setLaneletMap(
    state_.lanelet_map_ptr, routing_graph_ptr, traffic_rules_ptr);
  state_.predictor_vru->setLaneletMap(state_.lanelet_map_ptr);

  RCLCPP_DEBUG(node_->get_logger(), "[Map Based Prediction]: Map is loaded");
}

// ---------------------------------------------------------------------------
// ObjectsCallback
// ---------------------------------------------------------------------------

ObjectsCallback::ObjectsCallback(autoware::agnocast_wrapper::Node * node, NodeState & state)
: state_(state), transform_listener_(node)
{
  sub_traffic_signals_ =
    node->create_polling_subscriber<TrafficLightGroupArray>("/traffic_signals", rclcpp::QoS{1});
  stop_watch_ptr_ = std::make_unique<autoware_utils::StopWatch<std::chrono::milliseconds>>();
  stop_watch_ptr_->tic("cyclic_time");
  stop_watch_ptr_->tic("processing_time");
}

void ObjectsCallback::setObjectsPublisher(AUTOWARE_PUBLISHER_PTR(PredictedObjects) pub_objects)
{
  pub_objects_ = std::move(pub_objects);
}

void ObjectsCallback::setDebugMarkersPublisher(
  AUTOWARE_PUBLISHER_PTR(visualization_msgs::msg::MarkerArray) pub_debug_markers)
{
  pub_debug_markers_ = std::move(pub_debug_markers);
}

void ObjectsCallback::setDiagnostics(Diagnostics * diagnostics)
{
  diagnostics_ = diagnostics;
}

void ObjectsCallback::trafficSignalsCallback(
  const AUTOWARE_MESSAGE_CONST_SHARED_PTR(TrafficLightGroupArray) & msg)
{
  state_.predictor_vru->setTrafficSignal(*msg);
}

bool ObjectsCallback::hasValidHeaderStamp(
  const AUTOWARE_MESSAGE_CONST_SHARED_PTR(TrackedObjects) & in_objects)
{
  // Guard 1 (deterministic root cause): validate the raw header-stamp fields BEFORE anything builds
  // an rclcpp::Time from them. rclcpp::Time(stamp) throws "cannot store a negative time point" for
  // sec < 0, and every time conversion on the processing path (input stamp, tf lookup, history
  // pruning, diagnostics/publish stamp) derives from this one stamp - so validating it here, where
  // it cannot throw, pinpoints a corrupt input as the root cause and stops the crash at its source.
  const auto & stamp = in_objects->header.stamp;
  const bool is_valid = stamp.sec >= 0 && stamp.nanosec < 1000000000u;
  if (!is_valid) {
    RCLCPP_WARN(
      rclcpp::get_logger("map_based_prediction"),
      "[guard: input-stamp] dropped TrackedObjects with invalid header stamp (sec=%d nanosec=%u): "
      "sec must be >= 0 and nanosec < 1e9. frame_id='%s' n_objects=%zu. Likely a corrupt "
      "deserialization (see upstream CycloneDDS 'invalid data size' / typesupport errors).",
      stamp.sec, stamp.nanosec, in_objects->header.frame_id.c_str(), in_objects->objects.size());
  }
  return is_valid;
}

void ObjectsCallback::objectsCallback(
  const AUTOWARE_MESSAGE_CONST_SHARED_PTR(TrackedObjects) & in_objects)
{
  if (!hasValidHeaderStamp(in_objects)) return;

  // Guard 3 (last resort): the header stamp is validated above and the per-object loop is guarded
  // individually inside processObjects, so any exception reaching here comes from an unforeseen
  // unit. Keep the node alive and flag it as unattributed - if this ever fires, add a dedicated
  // small-unit guard for the reported stage rather than relying on this catch-all.
  try {
    processObjects(in_objects);
  } catch (const std::exception & e) {
    const auto & s = in_objects->header.stamp;
    RCLCPP_WARN(
      rclcpp::get_logger("map_based_prediction"),
      "[guard: unattributed] exception from an unguarded stage, message dropped to keep node "
      "alive: %s. Input header: stamp sec=%d nanosec=%u frame_id='%s' n_objects=%zu.",
      e.what(), s.sec, s.nanosec, in_objects->header.frame_id.c_str(), in_objects->objects.size());
  }
}

void ObjectsCallback::processObjects(
  const AUTOWARE_MESSAGE_CONST_SHARED_PTR(TrackedObjects) & in_objects)
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (state_.time_keeper) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *state_.time_keeper);

  const rclcpp::Time objects_time(in_objects->header.stamp);

  stop_watch_ptr_->toc("processing_time", true);

  {
    const auto msg = sub_traffic_signals_->take_data();
    if (msg) trafficSignalsCallback(msg);
  }

  if (!state_.lanelet_map_ptr) return;

  geometry_msgs::msg::TransformStamped::ConstSharedPtr world2map_transform;
  const bool is_object_not_in_map_frame = in_objects->header.frame_id != "map";
  if (is_object_not_in_map_frame) {
    world2map_transform = transform_listener_.get_transform(
      "map", in_objects->header.frame_id, objects_time, rclcpp::Duration::from_seconds(0.1));
    if (!world2map_transform) return;
  }

  const double objects_detected_time = objects_time.seconds();

  state_.predictor_vehicle->removeOldHistory(
    objects_detected_time, state_.params.object_buffer_time_length);
  state_.predictor_vru->removeOldKnownMatches(
    objects_detected_time, state_.params.object_buffer_time_length);

  auto output_msg = ALLOCATE_OUTPUT_MESSAGE_UNIQUE(pub_objects_);
  PredictedObjects & output = *output_msg;
  output.header = in_objects->header;
  output.header.frame_id = "map";

  visualization_msgs::msg::MarkerArray debug_markers;

  state_.predictor_vru->loadCurrentCrosswalkUsers(*in_objects);

  for (size_t object_index = 0; object_index < in_objects->objects.size(); ++object_index) {
    const auto & object = in_objects->objects[object_index];

    // Guard 2 (per-object root cause): one malformed object must not drop the whole frame nor abort
    // the node. Any exception here is attributed to this specific object - its index, uuid and label
    // are logged so the offending detection and the predictor that raised it can be traced - and
    // then skipped so the remaining objects are still predicted.
    try {
      TrackedObject transformed_object = object;

      if (is_object_not_in_map_frame) {
        geometry_msgs::msg::PoseStamped pose_in_map;
        geometry_msgs::msg::PoseStamped pose_orig;
        pose_orig.pose = object.kinematics.pose_with_covariance.pose;
        tf2::doTransform(pose_orig, pose_in_map, *world2map_transform);
        transformed_object.kinematics.pose_with_covariance.pose = pose_in_map.pose;
      }

      const auto & label_ =
        autoware::object_recognition_utils::getHighestProbLabel(transformed_object.classification);
      const auto label = utils::changeVRULabelForPrediction(label_, object, state_.lanelet_map_ptr);

      switch (label) {
        case ObjectClassification::PEDESTRIAN:
        case ObjectClassification::BICYCLE: {
          output.objects.emplace_back(
            state_.predictor_vru->predict(output.header, transformed_object));
          break;
        }
        case ObjectClassification::CAR:
        case ObjectClassification::BUS:
        case ObjectClassification::TRAILER:
        case ObjectClassification::MOTORCYCLE:
        case ObjectClassification::TRUCK: {
          const auto predicted_object_opt = state_.predictor_vehicle->predict(
            output.header, transformed_object, objects_detected_time,
            pub_debug_markers_ ? &debug_markers : nullptr);
          if (predicted_object_opt) output.objects.push_back(predicted_object_opt.value());
          break;
        }
        default: {
          auto predicted_unknown_object = utils::convertToPredictedObject(transformed_object);
          PredictedPath predicted_path = state_.path_generator->generatePathForNonVehicleObject(
            transformed_object, state_.params.prediction_time_horizon_unknown);
          predicted_path.confidence = 1.0;
          predicted_unknown_object.kinematics.predicted_paths.push_back(predicted_path);
          output.objects.push_back(predicted_unknown_object);
          break;
        }
      }
    } catch (const std::exception & e) {
      const auto label = autoware::object_recognition_utils::getHighestProbLabel(
        object.classification);
      RCLCPP_WARN(
        rclcpp::get_logger("map_based_prediction"),
        "[guard: predict-object] skipped object %zu/%zu (uuid=%s label=%d): %s",
        object_index, in_objects->objects.size(),
        autoware_utils::to_hex_string(object.object_id).c_str(), static_cast<int>(label),
        e.what());
      continue;
    }
  }

  if (state_.params.remember_lost_crosswalk_users) {
    PredictedObjects retrieved_objects = state_.predictor_vru->retrieveUndetectedObjects();
    output.objects.insert(
      output.objects.end(), retrieved_objects.objects.begin(), retrieved_objects.objects.end());
  }

  publish(std::move(output_msg), debug_markers);

  const auto processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
  const auto cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);

  if (diagnostics_) diagnostics_->update(output.header.stamp, processing_time_ms, cyclic_time_ms);
}

void ObjectsCallback::publish(
  AUTOWARE_MESSAGE_UNIQUE_PTR(PredictedObjects) output,
  const visualization_msgs::msg::MarkerArray & debug_markers) const
{
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (state_.time_keeper) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *state_.time_keeper);

  const auto stamp = output->header.stamp;
  pub_objects_->publish(std::move(output));
  if (diagnostics_) diagnostics_->publishIfSubscribed<PredictedObjects>(pub_objects_, stamp);
  if (pub_debug_markers_) {
    auto debug_markers_msg = ALLOCATE_OUTPUT_MESSAGE_UNIQUE(pub_debug_markers_);
    *debug_markers_msg = debug_markers;
    pub_debug_markers_->publish(std::move(debug_markers_msg));
  }
}

}  // namespace autoware::map_based_prediction
