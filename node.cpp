// Copyright 2020 Tier IV, Inc.
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

#include "autoware/behavior_velocity_planner/node.hpp"

#include <autoware/behavior_velocity_planner_common/utilization/path_utilization.hpp>
#include <autoware/motion_utils/trajectory/path_with_lane_id.hpp>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware/velocity_smoother/smoother/analytical_jerk_constrained_smoother/analytical_jerk_constrained_smoother.hpp>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_utils/ros/wait_for_param.hpp>
#include <autoware_utils/transform/transforms.hpp>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <lanelet2_routing/Route.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <string>
#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>
#endif

#include <pmu_analyzer.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace autoware::behavior_velocity_planner
{
namespace
{

autoware_planning_msgs::msg::Path to_path(
  const autoware_internal_planning_msgs::msg::PathWithLaneId & path_with_id)
{
  autoware_planning_msgs::msg::Path path;
  for (const auto & path_point : path_with_id.points) {
    path.points.push_back(path_point.point);
  }
  return path;
}
}  // namespace

BehaviorVelocityPlannerNode::BehaviorVelocityPlannerNode(const rclcpp::NodeOptions & node_options)
: Node("behavior_velocity_planner_node", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_),
  planner_data_(*this)
{
  using std::placeholders::_1;
  using std::placeholders::_2;

  // Trigger Subscriber
  trigger_sub_path_with_lane_id_ =
    this->create_subscription<autoware_internal_planning_msgs::msg::PathWithLaneId>(
      "~/input/path_with_lane_id", 1, std::bind(&BehaviorVelocityPlannerNode::onTrigger, this, _1));

  srv_load_plugin_ = create_service<autoware_internal_debug_msgs::srv::String>(
    "~/service/load_plugin", std::bind(&BehaviorVelocityPlannerNode::onLoadPlugin, this, _1, _2));
  srv_unload_plugin_ = create_service<autoware_internal_debug_msgs::srv::String>(
    "~/service/unload_plugin",
    std::bind(&BehaviorVelocityPlannerNode::onUnloadPlugin, this, _1, _2));

  // set velocity smoother param
  onParam();

  // Publishers
  path_pub_ = this->create_publisher<autoware_planning_msgs::msg::Path>("~/output/path", 1);
  debug_viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/path", 1);

  // Parameters
  forward_path_length_ = declare_parameter<double>("forward_path_length");
  backward_path_length_ = declare_parameter<double>("backward_path_length");
  behavior_output_path_interval_ = declare_parameter<double>("behavior_output_path_interval");
  planner_data_.stop_line_extend_length = declare_parameter<double>("stop_line_extend_length");

  // nearest search
  planner_data_.ego_nearest_dist_threshold =
    declare_parameter<double>("ego_nearest_dist_threshold");
  planner_data_.ego_nearest_yaw_threshold = declare_parameter<double>("ego_nearest_yaw_threshold");

  // is simulation or not
  planner_data_.is_simulation = declare_parameter<bool>("is_simulation");

  // Initialize PlannerManager
  for (const auto & name : declare_parameter<std::vector<std::string>>("launch_modules")) {
    // workaround: Since ROS 2 can't get empty list, launcher set [''] on the parameter.
    if (name == "") {
      break;
    }
    planner_manager_.launchScenePlugin(*this, name);
  }

  logger_configure_ = std::make_unique<autoware_utils::LoggerLevelConfigure>(this);
  published_time_publisher_ = std::make_unique<autoware_utils::PublishedTimePublisher>(this);
  std::string session_name = "behavior_velocity_planner";
  pmu_analyzer::ELAPSED_TIME_INIT(session_name);
}

BehaviorVelocityPlannerNode::~BehaviorVelocityPlannerNode()
{
  std::string session_name = "behavior_velocity_planner";

  pmu_analyzer::ELAPSED_TIME_CLOSE(session_name);
}

void BehaviorVelocityPlannerNode::onLoadPlugin(
  const autoware_internal_debug_msgs::srv::String::Request::SharedPtr request,
  [[maybe_unused]] const autoware_internal_debug_msgs::srv::String::Response::SharedPtr response)
{
  std::unique_lock<std::mutex> lk(mutex_);
  planner_manager_.launchScenePlugin(*this, request->data);
}

void BehaviorVelocityPlannerNode::onUnloadPlugin(
  const autoware_internal_debug_msgs::srv::String::Request::SharedPtr request,
  [[maybe_unused]] const autoware_internal_debug_msgs::srv::String::Response::SharedPtr response)
{
  std::unique_lock<std::mutex> lk(mutex_);
  planner_manager_.removeScenePlugin(*this, request->data);
}

void BehaviorVelocityPlannerNode::onParam()
{
  // Note(VRichardJP): mutex lock is not necessary as onParam is only called once in the
  // constructed. It would be required if it was a callback. std::lock_guard<std::mutex>
  // lock(mutex_);
  planner_data_.velocity_smoother_ =
    std::make_unique<autoware::velocity_smoother::AnalyticalJerkConstrainedSmoother>(*this);
  planner_data_.velocity_smoother_->setWheelBase(planner_data_.vehicle_info_.wheel_base_m);
}

void BehaviorVelocityPlannerNode::processNoGroundPointCloud(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(
      "map", msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
  } catch (tf2::TransformException & e) {
    RCLCPP_WARN(get_logger(), "no transform found for no_ground_pointcloud: %s", e.what());
    return;
  }

  pcl::PointCloud<pcl::PointXYZ> pc;
  pcl::fromROSMsg(*msg, pc);

  Eigen::Affine3f affine = tf2::transformToEigen(transform.transform).cast<float>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc_transformed(new pcl::PointCloud<pcl::PointXYZ>);
  if (!pc.empty()) {
    autoware_utils::transform_pointcloud(pc, *pc_transformed, affine);
  }
  if (pc_transformed && !pc_transformed->empty()) {
    int pi = 0;
    std::string session_name = "behavior_velocity_planner";
    for (const auto & point : *pc_transformed) {
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "no_ground_pointcloud_transformed_" + std::to_string(pi),
        {point.x, point.y, point.z});
      pi++;
    }
  }
  planner_data_.no_ground_pointcloud = pc_transformed;
}

void BehaviorVelocityPlannerNode::processOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  auto current_odometry = std::make_shared<geometry_msgs::msg::PoseStamped>();
  current_odometry->header = msg->header;
  current_odometry->pose = msg->pose.pose;
  planner_data_.current_odometry = current_odometry;
  std::string session_name = "behavior_velocity_planner";
  pmu_analyzer::VAR_LOG_VEC(
    session_name, "current_odometry_pose",
    {current_odometry->pose.position.x, current_odometry->pose.position.y});

  auto current_velocity = std::make_shared<geometry_msgs::msg::TwistStamped>();
  current_velocity->header = msg->header;
  current_velocity->twist = msg->twist.twist;
  planner_data_.current_velocity = current_velocity;
  pmu_analyzer::VAR_LOG_VEC(
    session_name, "current_velocity_twist_linear",
    {current_velocity->twist.linear.x, current_velocity->twist.linear.y,
     current_velocity->twist.linear.z});
  pmu_analyzer::VAR_LOG_VEC(
    session_name, "current_velocity_twist_angular",
    {current_velocity->twist.angular.x, current_velocity->twist.angular.y,
     current_velocity->twist.angular.z});
  pmu_analyzer::VAR_LOG_SINGLE(
    session_name, "velocity_buffer_old_size", planner_data_.velocity_buffer.size());

  // Add velocity to buffer
  planner_data_.velocity_buffer.push_front(*current_velocity);
  const rclcpp::Time now = this->now();
  while (!planner_data_.velocity_buffer.empty()) {
    // Check oldest data time
    const auto & s = planner_data_.velocity_buffer.back().header.stamp;
    const auto time_diff =
      now >= s ? now - s : rclcpp::Duration(0, 0);  // Note: negative time throws an exception.

    // Finish when oldest data is newer than threshold
    if (time_diff.seconds() <= PlannerData::velocity_buffer_time_sec) {
      break;
    }

    // Remove old data
    planner_data_.velocity_buffer.pop_back();
  }
  pmu_analyzer::VAR_LOG_SINGLE(
    session_name, "velocity_buffer_new_size", planner_data_.velocity_buffer.size());

  int vb_idx = 0;
  std::vector<double> old_vb;
  for (auto & vb : planner_data_.velocity_buffer) {
    // pmu_analyzer::VAR_LOG_SINGLE(
    //   session_name, "velocity_buffer_old_sec_" + std::to_string(vb_idx),
    //   vb.header.stamp.seconds());
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "velocity_buffer_new_twist_linear_" + std::to_string(vb_idx),
      {vb.twist.linear.x, vb.twist.linear.y, vb.twist.linear.z});
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "velocity_buffer_new_twist_angular_" + std::to_string(vb_idx),
      {vb.twist.angular.x, vb.twist.angular.y, vb.twist.angular.z});
  }
}

void BehaviorVelocityPlannerNode::processTrafficSignals(
  const autoware_perception_msgs::msg::TrafficLightGroupArray::ConstSharedPtr msg)
{
  std::string session_name = "behavior_velocity_planner";
  pmu_analyzer::VAR_LOG_SINGLE(
    session_name, "traffic_light_id_map_raw_old_size",
    planner_data_.traffic_light_id_map_raw_.size());
  pmu_analyzer::VAR_LOG_SINGLE(
    session_name, "traffic_light_id_map_last_observed_old_size",
    planner_data_.traffic_light_id_map_last_observed_.size());
  pmu_analyzer::VAR_LOG_SINGLE(
    session_name, "traffic_light_signal_count", msg->traffic_light_groups.size());

  // clear previous observation
  planner_data_.traffic_light_id_map_raw_.clear();
  const auto traffic_light_id_map_last_observed_old =
    planner_data_.traffic_light_id_map_last_observed_;
  planner_data_.traffic_light_id_map_last_observed_.clear();
  int sid = 0;
  for (const auto & signal : msg->traffic_light_groups) {
    TrafficSignalStamped traffic_signal;
    traffic_signal.stamp = msg->stamp;
    traffic_signal.signal = signal;
    pmu_analyzer::VAR_LOG_SINGLE(
      session_name, "traffic_light_signal_traffic_light_group_id_" + std::to_string(sid),
      signal.traffic_light_group_id);
    int eid = 0;
    for (auto & e : signal.elements) {
      pmu_analyzer::VAR_LOG_VEC(
        session_name,
        "traffic_light_signal_element_" + std::to_string(eid) + "_" + std::to_string(sid),
        {static_cast<double>(e.color), static_cast<double>(e.shape), static_cast<double>(e.status),
         e.confidence});
      eid++;
    }
    sid++;

    planner_data_.traffic_light_id_map_raw_[signal.traffic_light_group_id] = traffic_signal;
    const bool is_unknown_observation =
      std::any_of(signal.elements.begin(), signal.elements.end(), [](const auto & element) {
        return element.color == autoware_perception_msgs::msg::TrafficLightElement::UNKNOWN;
      });
    // if the observation is UNKNOWN and past observation is available, only update the timestamp
    // and keep the body of the info
    const auto old_data =
      traffic_light_id_map_last_observed_old.find(signal.traffic_light_group_id);
    if (is_unknown_observation && old_data != traffic_light_id_map_last_observed_old.end()) {
      // copy last observation
      planner_data_.traffic_light_id_map_last_observed_[signal.traffic_light_group_id] =
        old_data->second;
      // update timestamp
      planner_data_.traffic_light_id_map_last_observed_[signal.traffic_light_group_id].stamp =
        msg->stamp;
    } else {
      // if (1)the observation is not UNKNOWN or (2)the very first observation is UNKNOWN
      planner_data_.traffic_light_id_map_last_observed_[signal.traffic_light_group_id] =
        traffic_signal;
    }
  }
}

bool BehaviorVelocityPlannerNode::processData(rclcpp::Clock clock)
{
  std::string session_name = "behavior_velocity_planner";

  bool is_ready = true;
  const auto & logData = [&clock, this](const std::string & data_type) {
    std::string msg = "Waiting for " + data_type + " data";
    RCLCPP_INFO_THROTTLE(get_logger(), clock, logger_throttle_interval, "%s", msg.c_str());
  };

  const auto & getData = [&logData](auto & dest, auto & sub, const std::string & data_type = "") {
    const auto temp = sub.take_data();
    if (temp) {
      dest = temp;
      return true;
    }
    if (!data_type.empty()) logData(data_type);
    return false;
  };

  is_ready &= getData(planner_data_.current_acceleration, sub_acceleration_, "acceleration");
  is_ready &= getData(planner_data_.predicted_objects, sub_predicted_objects_, "predicted_objects");
  is_ready &= getData(planner_data_.occupancy_grid, sub_occupancy_grid_, "occupancy_grid");

  const auto odometry = sub_vehicle_odometry_.take_data();
  if (odometry) {
    processOdometry(odometry);
  } else {
    logData("odometry");
    is_ready = false;
  }

  const auto no_ground_pointcloud = sub_no_ground_pointcloud_.take_data();
  if (no_ground_pointcloud) {
    processNoGroundPointCloud(no_ground_pointcloud);
    pmu_analyzer::VAR_LOG_SINGLE(
      session_name, "no_ground_pointcloud_size", no_ground_pointcloud->width);
  } else {
    logData("pointcloud");
    is_ready = false;
  }

  const auto map_data = sub_lanelet_map_.take_data();
  if (map_data) {
    planner_data_.route_handler_ = std::make_shared<route_handler::RouteHandler>(*map_data);
    pmu_analyzer::VAR_LOG_SINGLE(session_name, "map_data_size", map_data->data.size());
  }

  // planner_data_.external_velocity_limit is std::optional type variable.
  const auto external_velocity_limit = sub_external_velocity_limit_.take_data();
  if (external_velocity_limit) {
    planner_data_.external_velocity_limit = *external_velocity_limit;
    pmu_analyzer::VAR_LOG_SINGLE(
      session_name, "external_velocity_limit_max_velocity",
      planner_data_.external_velocity_limit->max_velocity);
  }

  const auto traffic_signals = sub_traffic_signals_.take_data();
  if (traffic_signals) processTrafficSignals(traffic_signals);

  if (is_ready) {
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "current_acceleration_accel_linear",
      {planner_data_.current_acceleration->accel.accel.linear.x,
       planner_data_.current_acceleration->accel.accel.linear.y,
       planner_data_.current_acceleration->accel.accel.linear.z});
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "current_acceleration_accel_angular",
      {planner_data_.current_acceleration->accel.accel.angular.x,
       planner_data_.current_acceleration->accel.accel.angular.y,
       planner_data_.current_acceleration->accel.accel.angular.z});
    pmu_analyzer::VAR_LOG_SINGLE(
      session_name, "predicted_objects_size", planner_data_.predicted_objects->objects.size());
    int oid = 0;
    for (auto & obj : planner_data_.predicted_objects->objects) {
      pmu_analyzer::VAR_LOG_SINGLE(
        session_name, "predicted_object_existence_probability_" + std::to_string(oid),
        obj.existence_probability);
      int cid = 0;
      for (auto & cls : obj.classification) {
        pmu_analyzer::VAR_LOG_SINGLE(
          session_name, "predicted_object_label_" + std::to_string(cid) + "_" + std::to_string(oid),
          cls.label);
        pmu_analyzer::VAR_LOG_SINGLE(
          session_name,
          "predicted_object_probability_" + std::to_string(cid) + "_" + std::to_string(oid),
          cls.probability);
        cid++;
      }
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "predicted_object_kinematics_pose_position_" + std::to_string(oid),
        {obj.kinematics.initial_pose_with_covariance.pose.position.x,
         obj.kinematics.initial_pose_with_covariance.pose.position.y,
         obj.kinematics.initial_pose_with_covariance.pose.position.z});
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "predicted_object_kinematics_pose_orientation_" + std::to_string(oid),
        {obj.kinematics.initial_pose_with_covariance.pose.orientation.x,
         obj.kinematics.initial_pose_with_covariance.pose.orientation.y,
         obj.kinematics.initial_pose_with_covariance.pose.orientation.z,
         obj.kinematics.initial_pose_with_covariance.pose.orientation.w});

      // initial_twist_with_covariance
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "predicted_object_kinematics_twist_linear_" + std::to_string(oid),
        {obj.kinematics.initial_twist_with_covariance.twist.linear.x,
         obj.kinematics.initial_twist_with_covariance.twist.linear.y,
         obj.kinematics.initial_twist_with_covariance.twist.linear.z});
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "predicted_object_kinematics_twist_angular_" + std::to_string(oid),
        {obj.kinematics.initial_twist_with_covariance.twist.angular.x,
         obj.kinematics.initial_twist_with_covariance.twist.angular.y,
         obj.kinematics.initial_twist_with_covariance.twist.angular.z});
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "predicted_object_kinematics_accel_linear_" + std::to_string(oid),
        {obj.kinematics.initial_acceleration_with_covariance.accel.linear.x,
         obj.kinematics.initial_acceleration_with_covariance.accel.linear.y,
         obj.kinematics.initial_acceleration_with_covariance.accel.linear.z});
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "predicted_object_kinematics_accel_ang_x_" + std::to_string(oid),
        {obj.kinematics.initial_acceleration_with_covariance.accel.angular.x,
         obj.kinematics.initial_acceleration_with_covariance.accel.angular.y,
         obj.kinematics.initial_acceleration_with_covariance.accel.angular.z});
      pmu_analyzer::VAR_LOG_SINGLE(
        session_name, "predicted_object_shape_type_" + std::to_string(oid), obj.shape.type);
      pmu_analyzer::VAR_LOG_VEC(
        session_name, "predicted_object_shape_dimensions_" + std::to_string(oid),
        {obj.shape.dimensions.x, obj.shape.dimensions.y, obj.shape.dimensions.z});

      for (size_t i = 0; i < obj.shape.footprint.points.size(); ++i) {
        const auto & pt = obj.shape.footprint.points[i];
        pmu_analyzer::VAR_LOG_VEC(
          session_name,
          "predicted_object_shape_footprint_p" + std::to_string(i) + "_" + std::to_string(oid),
          {pt.x, pt.y, pt.z});
      }
      oid++;
    }
    // pmu_analyzer::VAR_LOG_SINGLE(
    //   session_name, "occupancy_grid_width", planner_data_.occupancy_grid->info.width);
    // pmu_analyzer::VAR_LOG_SINGLE(
    //   session_name, "occupancy_grid_height", planner_data_.occupancy_grid->info.height);
    // pmu_analyzer::VAR_LOG_SINGLE(
    //   session_name, "occupancy_grid_resolution", planner_data_.occupancy_grid->info.resolution);
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "occupancy_grid_origin",
      {planner_data_.occupancy_grid->info.origin.position.x,
       planner_data_.occupancy_grid->info.origin.position.y,
       planner_data_.occupancy_grid->info.origin.position.z});
    double occupancy_grid_ave = 0;
    // std::vector<double> ds;
    for (size_t i = 0; i < planner_data_.occupancy_grid->data.size(); i++) {
      occupancy_grid_ave += planner_data_.occupancy_grid->data[i];
    }
    pmu_analyzer::VAR_LOG_SINGLE(session_name, "occupancy_grid_data_average", occupancy_grid_ave);
  }
  return is_ready;
}

// NOTE: argument planner_data must not be referenced for multithreading
bool BehaviorVelocityPlannerNode::isDataReady(rclcpp::Clock clock)
{
  if (!planner_data_.velocity_smoother_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), clock, logger_throttle_interval,
      "Waiting for the initialization of velocity smoother");
    return false;
  }

  return processData(clock);
}

void BehaviorVelocityPlannerNode::onTrigger(
  const autoware_internal_planning_msgs::msg::PathWithLaneId::ConstSharedPtr input_path_msg)
{
  std::string session_name = "behavior_velocity_planner";
  std::unique_lock<std::mutex> lk(mutex_);
  pmu_analyzer::ELAPSED_TIME_TIMESTAMP(
    session_name, 0 /* part index */, true /* is first in this loop? */,
    0 /* data (forloop index or data size etc..) */);
  if (!isDataReady(*get_clock())) {
    return;
  }

  // Load map and check route handler
  if (!planner_data_.route_handler_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), logger_throttle_interval,
      "Waiting for the initialization of route_handler");
    return;
  }

  if (input_path_msg->points.empty()) {
    return;
  }
  pmu_analyzer::VAR_LOG_SINGLE(
    session_name, "input_path_msg_point_num", input_path_msg->points.size());
  // output all points
  int pt_idx = 0;
  auto lanelet_map = planner_data_.route_handler_->getLaneletMapPtr();
  for (auto & p : input_path_msg->points) {
    auto & pt = p.point;
    std::vector<double> vec_pt_pos = {pt.pose.position.x, pt.pose.position.y, pt.pose.position.z},
                        vec_pt_ori =
                          {pt.pose.orientation.x, pt.pose.orientation.y, pt.pose.orientation.z,
                           pt.pose.orientation.w},
                        vec_vel_mps = {pt.longitudinal_velocity_mps, pt.lateral_velocity_mps};
    std::vector<std::string> vec_lane_subtype, vec_lane_turn_direction, vec_lane_reg_type;
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "input_path_msg_point_position_" + std::to_string(pt_idx), vec_pt_pos);
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "input_path_msg_point_orientation_" + std::to_string(pt_idx), vec_pt_ori);
    pmu_analyzer::VAR_LOG_VEC(
      session_name, "input_path_msg_point_velocity_mps_" + std::to_string(pt_idx), vec_vel_mps);
    pmu_analyzer::VAR_LOG_SINGLE(
      session_name, "input_path_msg_heading_rate_rps_" + std::to_string(pt_idx),
      pt.heading_rate_rps);
    pmu_analyzer::VAR_LOG_SINGLE(
      session_name, "input_path_msg_heading_is_final_" + std::to_string(pt_idx),
      pt.is_final ? 1.0 : 0.0);
    for (auto & l : p.lane_ids) {
      const auto & lanelet = lanelet_map->laneletLayer.get(l);
      std::string subtype = lanelet.attributeOr("subtype", "none");
      std::string turn_direction = lanelet.attributeOr("turn_direction", "none");
      vec_lane_subtype.push_back(subtype);
      vec_lane_turn_direction.push_back(turn_direction);
      for (const auto & reg_elem : lanelet.regulatoryElements()) {
        std::string reg_elem_type = reg_elem->attributeOr("subtype", "none");
        vec_lane_reg_type.push_back(reg_elem_type);
      }
    }
    pmu_analyzer::VAR_LOG_STRING(
      session_name, "input_path_msg_lane_subtype_" + std::to_string(pt_idx), vec_lane_subtype);
    pmu_analyzer::VAR_LOG_STRING(
      session_name, "input_path_msg_lane_turn_direction_" + std::to_string(pt_idx),
      vec_lane_turn_direction);
    pmu_analyzer::VAR_LOG_STRING(
      session_name, "input_path_msg_lane_regulatory_type_" + std::to_string(pt_idx),
      vec_lane_reg_type);
    pt_idx++;
  }
  pmu_analyzer::ELAPSED_TIME_TIMESTAMP(session_name, 1 /* part index */, false, 0);
  const autoware_planning_msgs::msg::Path output_path_msg =
    generatePath(input_path_msg, planner_data_);
  pmu_analyzer::ELAPSED_TIME_TIMESTAMP(session_name, 2 /* part index */, false, 0);
  lk.unlock();

  path_pub_->publish(output_path_msg);
  published_time_publisher_->publish_if_subscribed(path_pub_, output_path_msg.header.stamp);

  if (debug_viz_pub_->get_subscription_count() > 0) {
    publishDebugMarker(output_path_msg);
  }
  pmu_analyzer::ELAPSED_TIME_TIMESTAMP(session_name, 3 /* part index */, false, 0);
}

autoware_planning_msgs::msg::Path BehaviorVelocityPlannerNode::generatePath(
  const autoware_internal_planning_msgs::msg::PathWithLaneId::ConstSharedPtr input_path_msg,
  const PlannerData & planner_data)
{
  autoware_planning_msgs::msg::Path output_path_msg;

  // TODO(someone): support backward path
  const auto is_driving_forward = autoware::motion_utils::isDrivingForward(input_path_msg->points);
  is_driving_forward_ = is_driving_forward ? is_driving_forward.value() : is_driving_forward_;
  std::string session_name = "behavior_velocity_planner";

  pmu_analyzer::VAR_LOG_SINGLE(session_name, "is_driving_forward_", is_driving_forward_);
  if (!is_driving_forward_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), logger_throttle_interval,
      "Backward path is NOT supported. just converting path_with_lane_id to path");
    output_path_msg = to_path(*input_path_msg);
    output_path_msg.header.frame_id = "map";
    output_path_msg.header.stamp = input_path_msg->header.stamp;
    output_path_msg.left_bound = input_path_msg->left_bound;
    output_path_msg.right_bound = input_path_msg->right_bound;
    return output_path_msg;
  }

  // Plan path velocity
  const auto velocity_planned_path = planner_manager_.planPathVelocity(
    std::make_shared<const PlannerData>(planner_data), *input_path_msg);

  // screening
  const auto filtered_path =
    autoware::behavior_velocity_planner::filterLitterPathPoint(to_path(velocity_planned_path));

  // interpolation
  const auto interpolated_path_msg = autoware::behavior_velocity_planner::interpolatePath(
    filtered_path, forward_path_length_, behavior_output_path_interval_);

  // check stop point
  output_path_msg = autoware::behavior_velocity_planner::filterStopPathPoint(interpolated_path_msg);

  output_path_msg.header.frame_id = "map";
  output_path_msg.header.stamp = input_path_msg->header.stamp;

  // TODO(someone): This must be updated in each scene module, but copy from input message for now.
  output_path_msg.left_bound = input_path_msg->left_bound;
  output_path_msg.right_bound = input_path_msg->right_bound;

  return output_path_msg;
}

void BehaviorVelocityPlannerNode::publishDebugMarker(const autoware_planning_msgs::msg::Path & path)
{
  visualization_msgs::msg::MarkerArray output_msg;
  for (size_t i = 0; i < path.points.size(); ++i) {
    visualization_msgs::msg::Marker marker;
    marker.header = path.header;
    marker.id = i;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.pose = path.points.at(i).pose;
    marker.scale.y = marker.scale.z = 0.05;
    marker.scale.x = 0.25;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    output_msg.markers.push_back(marker);
  }
  debug_viz_pub_->publish(output_msg);
}
}  // namespace autoware::behavior_velocity_planner

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::behavior_velocity_planner::BehaviorVelocityPlannerNode)
