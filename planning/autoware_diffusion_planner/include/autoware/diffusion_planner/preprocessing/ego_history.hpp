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

#ifndef AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__EGO_HISTORY_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__EGO_HISTORY_HPP_

#include "autoware/diffusion_planner/constants.hpp"
#include "autoware/diffusion_planner/dimensions.hpp"

#include <Eigen/Core>
#include <rclcpp/time.hpp>

#include <nav_msgs/msg/odometry.hpp>

#include <deque>
#include <memory>
#include <utility>
#include <vector>

namespace autoware::diffusion_planner::preprocess
{

/// @brief Owns the ego odometry buffer, kept monotonic oldest-to-newest and pruned to a time
/// window.
class EgoHistory
{
public:
  // Buffer spans the ego-history horizon (INPUT_T steps) + ego-object gap + 0.5 s headroom.
  static constexpr double DEFAULT_BUFFER_WINDOW_S =
    INPUT_T * constants::PREDICTION_TIME_STEP_S + constants::MAX_EGO_OBJECT_TIME_DIFF_S + 0.5;

  explicit EgoHistory(double buffer_window_s = DEFAULT_BUFFER_WINDOW_S);

  /// @brief Ingest raw odometry, keep the buffer monotonic, and prune to the time window.
  void update(const std::vector<std::shared_ptr<const nav_msgs::msg::Odometry>> & ego_states);

  bool empty() const { return buffer_.empty(); }

  /// @brief Stamp of the newest buffered sample. Precondition: !empty().
  rclcpp::Time newest_stamp() const;

  /// @brief Ego state at frame_time (interpolated when enabled and in range, else nearest sample).
  ///        Returns the state and its absolute time offset from frame_time [s]. Precondition:
  ///        !empty().
  std::pair<nav_msgs::msg::Odometry, double> select_state(
    const rclcpp::Time & frame_time, bool use_time_interpolation) const;

  /// @brief Past ego trajectory in the current ego frame, resampled at PREDICTION_TIME_STEP_S back
  ///        from reference_time. Flat [x, y, cos_yaw, sin_yaw] per step, length
  ///        EGO_HISTORY_SHAPE[1] * 4.
  std::vector<float> to_agent_past(
    const Eigen::Matrix4d & map_to_ego_transform, const rclcpp::Time & reference_time,
    bool use_time_interpolation, bool shift_x, double base_link_to_center) const;

  // Static primitives over an explicit oldest-to-newest buffer.

  static std::pair<nav_msgs::msg::Odometry, double> select_ego_state(
    const std::deque<nav_msgs::msg::Odometry> & buffer, const rclcpp::Time & frame_time,
    bool use_time_interpolation);

  static std::vector<float> create_ego_agent_past(
    const std::deque<nav_msgs::msg::Odometry> & buffer, size_t num_timesteps,
    const Eigen::Matrix4d & map_to_ego_transform, const rclcpp::Time & reference_time,
    bool use_time_interpolation);

private:
  std::deque<nav_msgs::msg::Odometry> buffer_;
  double buffer_window_s_;
};

}  // namespace autoware::diffusion_planner::preprocess

#endif  // AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__EGO_HISTORY_HPP_
