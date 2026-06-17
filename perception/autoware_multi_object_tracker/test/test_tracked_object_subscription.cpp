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

#include "autoware/multi_object_tracker/object_model/classes.hpp"
#include "autoware/multi_object_tracker/tracker/trackers/polygon_tracker.hpp"
#include "autoware/multi_object_tracker/types.hpp"
#include "autoware/multi_object_tracker/uncertainty/uncertainty_processor.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/shape.hpp>
#include <autoware_perception_msgs/msg/tracked_object.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

namespace
{
namespace mot = autoware::multi_object_tracker;

rclcpp::Time baseTime()
{
  return rclcpp::Time(1000000000LL, RCL_ROS_TIME);
}

unique_identifier_msgs::msg::UUID makeUuid(const uint8_t seed)
{
  unique_identifier_msgs::msg::UUID uuid;
  for (size_t i = 0; i < uuid.uuid.size(); ++i) {
    uuid.uuid[i] = static_cast<uint8_t>(seed + i);
  }
  return uuid;
}

bool uuidEqual(
  const unique_identifier_msgs::msg::UUID & a, const unique_identifier_msgs::msg::UUID & b)
{
  return mot::types::UUIDEqual{}(a, b);
}

// A minimal polygon DynamicObject + tracker so the binding API can be exercised on a real Tracker.
std::shared_ptr<mot::Tracker> makePolygonTracker(const rclcpp::Time & time)
{
  mot::types::DynamicObject obj;
  obj.time = time;
  obj.classification = {{mot::classes::Label::UNKNOWN, 1.0F}};
  obj.pose.orientation.w = 1.0;
  obj.pose_covariance.fill(0.0);
  obj.twist_covariance.fill(0.0);
  obj.kinematics.orientation_availability = mot::types::OrientationAvailability::AVAILABLE;
  obj.existence_probability = 0.9;
  obj.channel_index = 0;
  obj.shape.type = autoware_perception_msgs::msg::Shape::POLYGON;
  obj.shape.dimensions.z = 1.5;
  for (const auto & [px, py] : std::vector<std::pair<float, float>>{
         {1.0F, 1.0F}, {-1.0F, 1.0F}, {-1.0F, -1.0F}, {1.0F, -1.0F}}) {
    geometry_msgs::msg::Point32 point;
    point.x = px;
    point.y = py;
    obj.shape.footprint.points.push_back(point);
  }
  obj.area = 4.0;

  mot::types::DynamicObjectList list;
  list.channel_index = obj.channel_index;
  list.objects = {obj};
  const auto with_uncertainty = mot::uncertainty::modelUncertainty(list).objects.front();

  mot::PolygonTrackerConfig polygon_config;
  polygon_config.enable_velocity_estimation = false;
  return std::make_shared<mot::PolygonTracker>(time, with_uncertainty, polygon_config);
}
}  // namespace

// ---------------------------------------------------------------------------
// toDynamicObject(TrackedObject)
// ---------------------------------------------------------------------------
TEST(ToDynamicObjectTracked, CarriesSourceUuidAndKinematics)
{
  autoware_perception_msgs::msg::TrackedObject trk;
  trk.object_id = makeUuid(7);
  trk.existence_probability = 0.5F;
  {
    autoware_perception_msgs::msg::ObjectClassification c;
    c.label = autoware_perception_msgs::msg::ObjectClassification::CAR;
    c.probability = 1.0F;
    trk.classification.push_back(c);
  }
  trk.kinematics.pose_with_covariance.pose.position.x = 12.0;
  trk.kinematics.pose_with_covariance.pose.position.y = -3.0;
  trk.kinematics.twist_with_covariance.twist.linear.x = 4.0;
  trk.kinematics.orientation_availability =
    autoware_perception_msgs::msg::TrackedObjectKinematics::AVAILABLE;
  trk.kinematics.is_stationary = true;
  trk.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  trk.shape.dimensions.x = 4.0;
  trk.shape.dimensions.y = 2.0;

  constexpr uint kChannel = 3;
  const auto dyn = mot::types::toDynamicObject(trk, kChannel);

  // Upstream identity is carried as source_uuid; the internal uuid is freshly generated.
  EXPECT_TRUE(dyn.has_source_uuid);
  EXPECT_TRUE(uuidEqual(dyn.source_uuid, trk.object_id));
  EXPECT_FALSE(uuidEqual(dyn.uuid, trk.object_id));

  EXPECT_EQ(dyn.channel_index, kChannel);
  EXPECT_FLOAT_EQ(dyn.existence_probability, 0.5F);
  ASSERT_EQ(dyn.classification.size(), 1U);
  EXPECT_EQ(dyn.classification.front().label, mot::classes::Label::CAR);

  EXPECT_DOUBLE_EQ(dyn.pose.position.x, 12.0);
  EXPECT_DOUBLE_EQ(dyn.pose.position.y, -3.0);
  EXPECT_DOUBLE_EQ(dyn.twist.linear.x, 4.0);

  // TrackedObjectKinematics has no has_* flags: a tracked object always supplies them.
  EXPECT_TRUE(dyn.kinematics.has_position_covariance);
  EXPECT_TRUE(dyn.kinematics.has_twist);
  EXPECT_TRUE(dyn.kinematics.has_twist_covariance);
  EXPECT_TRUE(dyn.kinematics.is_stationary);
  EXPECT_EQ(
    dyn.kinematics.orientation_availability, mot::types::OrientationAvailability::AVAILABLE);

  EXPECT_EQ(dyn.shape.type, autoware_perception_msgs::msg::Shape::BOUNDING_BOX);
}

TEST(ToDynamicObjectTracked, ListConversionStampsEveryObject)
{
  autoware_perception_msgs::msg::TrackedObjects trks;
  trks.header.frame_id = "map";
  for (uint8_t i = 0; i < 3; ++i) {
    autoware_perception_msgs::msg::TrackedObject trk;
    trk.object_id = makeUuid(static_cast<uint8_t>(10 * (i + 1)));
    trks.objects.push_back(trk);
  }

  const auto list = mot::types::toDynamicObjectList(trks, 1);
  ASSERT_EQ(list.objects.size(), 3U);
  for (const auto & obj : list.objects) {
    EXPECT_TRUE(obj.has_source_uuid);
    EXPECT_EQ(obj.channel_index, 1U);
    // uuid index is built and resolvable by the freshly generated internal uuid.
    EXPECT_TRUE(list.getObjectIndexByUuid(obj.uuid).has_value());
  }
}

// ---------------------------------------------------------------------------
// Tracker source-UUID bindings
// ---------------------------------------------------------------------------
TEST(TrackerSourceBinding, BindRefreshAndCrossSource)
{
  const auto t0 = baseTime();
  auto tracker = makePolygonTracker(t0);
  EXPECT_TRUE(tracker->getSourceBindings().empty());

  const auto uuid_a = makeUuid(1);
  tracker->bindSource(2, uuid_a, t0, 2.0);
  ASSERT_EQ(tracker->getSourceBindings().size(), 1U);
  EXPECT_EQ(tracker->getSourceBindings().front().channel, 2U);
  EXPECT_TRUE(uuidEqual(tracker->getSourceBindings().front().uuid, uuid_a));

  // Re-confirming the same upstream id on the same channel refreshes last_seen, no new entry.
  const auto t1 = t0 + rclcpp::Duration::from_seconds(0.5);
  tracker->bindSource(2, uuid_a, t1, 2.0);
  ASSERT_EQ(tracker->getSourceBindings().size(), 1U);
  EXPECT_EQ(tracker->getSourceBindings().front().last_seen.nanoseconds(), t1.nanoseconds());

  // A different channel adds a second binding (cross-source fusion onto one tracker).
  const auto uuid_c = makeUuid(50);
  tracker->bindSource(5, uuid_c, t1, 2.0);
  EXPECT_EQ(tracker->getSourceBindings().size(), 2U);
}

TEST(TrackerSourceBinding, FreshSlotIsNotStolenButStaleSlotIsOverwritten)
{
  const auto t0 = baseTime();
  auto tracker = makePolygonTracker(t0);

  const auto uuid_a = makeUuid(1);
  const auto uuid_b = makeUuid(100);
  tracker->bindSource(2, uuid_a, t0, 2.0);

  // While the binding is still fresh, a different upstream id must NOT steal the slot
  // (UUID binding beats geometry).
  const auto t_fresh = t0 + rclcpp::Duration::from_seconds(1.0);
  tracker->bindSource(2, uuid_b, t_fresh, 2.0);
  ASSERT_EQ(tracker->getSourceBindings().size(), 1U);
  EXPECT_TRUE(uuidEqual(tracker->getSourceBindings().front().uuid, uuid_a));

  // Once the slot is stale (last_seen + timeout exceeded), a new id may overwrite it.
  const auto t_stale = t0 + rclcpp::Duration::from_seconds(2.5);
  tracker->bindSource(2, uuid_b, t_stale, 2.0);
  ASSERT_EQ(tracker->getSourceBindings().size(), 1U);
  EXPECT_TRUE(uuidEqual(tracker->getSourceBindings().front().uuid, uuid_b));
}

TEST(TrackerSourceBinding, PruneRemovesOnlyStaleBindings)
{
  const auto t0 = baseTime();
  auto tracker = makePolygonTracker(t0);

  // Two channels, both seen at t0 with a 2 s timeout.
  tracker->bindSource(2, makeUuid(1), t0, 2.0);
  const auto t_mid = t0 + rclcpp::Duration::from_seconds(2.0);
  tracker->bindSource(5, makeUuid(50), t_mid, 2.0);  // refreshed later
  ASSERT_EQ(tracker->getSourceBindings().size(), 2U);

  // At t0 + 3 s: channel 2 (last seen t0) is stale; channel 5 (last seen t0+2) is still fresh.
  tracker->pruneStaleSourceBindings(t0 + rclcpp::Duration::from_seconds(3.0));
  ASSERT_EQ(tracker->getSourceBindings().size(), 1U);
  EXPECT_EQ(tracker->getSourceBindings().front().channel, 5U);

  // Far in the future everything ages out — but the tracker object itself is untouched.
  tracker->pruneStaleSourceBindings(t0 + rclcpp::Duration::from_seconds(100.0));
  EXPECT_TRUE(tracker->getSourceBindings().empty());
}
