#include "control_pkg/control_node.hpp"
#include "control_pkg/msg/trajectory.hpp"
#include "control_pkg/msg/trajectory_point.hpp"
#include "control_pkg/msg/localization.hpp"
#include "control_pkg/msg/chassis.hpp"
#include "control_pkg/msg/control_command.hpp"
#include "control_pkg/msg/debug_info.hpp"
#include <cmath>
#include <limits>

namespace control_pkg
{

ControlNode::ControlNode(const std::string & node_name) : Node(node_name)
{
  // 声明参数
  this->declare_parameter("control_frequency", 50.0);
  this->declare_parameter("wheelbase", 2.8);
  this->declare_parameter("prediction_horizon", 20);
  this->declare_parameter("dt", 0.05);
  this->declare_parameter("max_steering_angle", 0.6);
  this->declare_parameter("max_steering_rate", 0.5);
  this->declare_parameter("max_acceleration", 3.0);
  this->declare_parameter("max_deceleration", -5.0);
  this->declare_parameter("max_msg_age_ms", 500.0);
  this->declare_parameter("max_lateral_error", 2.0);
  this->declare_parameter("max_heading_error", 1.0);

  control_frequency_ = this->get_parameter("control_frequency").as_double();

  // 配置 MPC
  MPCController::Config mpc_config;
  mpc_config.wheelbase = this->get_parameter("wheelbase").as_double();
  mpc_config.prediction_horizon = this->get_parameter("prediction_horizon").as_int();
  mpc_config.dt = this->get_parameter("dt").as_double();
  mpc_config.max_steering_angle = this->get_parameter("max_steering_angle").as_double();
  mpc_config.max_steering_rate = this->get_parameter("max_steering_rate").as_double();
  mpc_config.max_acceleration = this->get_parameter("max_acceleration").as_double();
  mpc_config.max_deceleration = this->get_parameter("max_deceleration").as_double();
  mpc_controller_ = std::make_unique<MPCController>(mpc_config);

  // 配置异常检测器
  anomaly_detector_ = std::make_unique<AnomalyDetector>(
      this->get_parameter("max_msg_age_ms").as_double(),
      this->get_parameter("max_lateral_error").as_double(),
      this->get_parameter("max_heading_error").as_double());

  // 发布者
  control_pub_ = this->create_publisher<msg::ControlCommand>("control_command", 10);
  debug_pub_ = this->create_publisher<msg::DebugInfo>("debug_info", 10);

  // 订阅者 (主线程 Spin 负责收消息)
  traj_sub_ = this->create_subscription<msg::Trajectory>(
      "planning_trajectory", 10,
      std::bind(&ControlNode::TrajectoryCallback, this, std::placeholders::_1));
  loc_sub_ = this->create_subscription<msg::Localization>(
      "localization", 10,
      std::bind(&ControlNode::LocalizationCallback, this, std::placeholders::_1));
  chassis_sub_ = this->create_subscription<msg::Chassis>(
      "chassis", 10,
      std::bind(&ControlNode::ChassisCallback, this, std::placeholders::_1));

  // 启动专用控制线程
  cycle_period_ = std::chrono::nanoseconds(
      static_cast<int64_t>(1e9 / control_frequency_));
  control_thread_ = std::thread(&ControlNode::ControlLoop, this);

  RCLCPP_INFO(this->get_logger(),
              "ControlNode initialized: %.1f Hz control thread, wheelbase=%.1f",
              control_frequency_, mpc_config.wheelbase);
}

ControlNode::~ControlNode()
{
  running_ = false;
  if (control_thread_.joinable()) control_thread_.join();
}

// ===== 订阅回调 (极轻量，仅做缓存) =====

void ControlNode::TrajectoryCallback(const std::shared_ptr<msg::Trajectory> msg)
{
  latest_trajectory_ = msg;
  traj_received_ = true;
}

void ControlNode::LocalizationCallback(const std::shared_ptr<msg::Localization> msg)
{
  latest_localization_ = msg;
  loc_received_ = true;
}

void ControlNode::ChassisCallback(const std::shared_ptr<msg::Chassis> msg)
{
  latest_chassis_ = msg;
  chassis_received_ = true;
}

// ===== 工具函数 =====

std::vector<TrajectoryPoint> ControlNode::ConvertTrajectory(
    const std::shared_ptr<msg::Trajectory> & msg) const
{
  std::vector<TrajectoryPoint> result;
  if (!msg) return result;
  result.reserve(msg->points.size());
  for (const auto & pt : msg->points) {
    result.push_back({pt.x, pt.y, pt.z, pt.yaw, pt.velocity, pt.curvature, pt.relative_time});
  }
  return result;
}

int ControlNode::FindClosestPointIndex(double x, double y,
                                       const std::vector<TrajectoryPoint> & traj) const
{
  int closest = 0;
  double min_d = std::numeric_limits<double>::max();
  for (size_t i = 0; i < traj.size(); ++i) {
    double dx = traj[i].x - x, dy = traj[i].y - y;
    double d = dx * dx + dy * dy;
    if (d < min_d) { min_d = d; closest = static_cast<int>(i); }
  }
  return closest;
}

double ControlNode::ComputeLateralError(double x, double y, double heading,
                                        const TrajectoryPoint & ref) const
{
  (void)heading;
  double dx = x - ref.x, dy = y - ref.y;
  return -std::sin(ref.yaw) * dx + std::cos(ref.yaw) * dy;
}

double ControlNode::ComputeHeadingError(double heading,
                                        const TrajectoryPoint & ref) const
{
  double e = heading - ref.yaw;
  e = std::fmod(e + M_PI, 2.0 * M_PI);
  if (e < 0) e += 2.0 * M_PI;
  return e - M_PI;
}

StateVec ControlNode::GetCurrentState() const
{
  StateVec s = StateVec::Zero();
  if (latest_localization_) {
    s(0) = latest_localization_->x;
    s(1) = latest_localization_->y;
    s(2) = latest_localization_->yaw;
    s(3) = latest_chassis_ ? latest_chassis_->speed
           : std::sqrt(latest_localization_->velocity_x * latest_localization_->velocity_x +
                       latest_localization_->velocity_y * latest_localization_->velocity_y);
  }
  return s;
}

// ===== 控制循环 (专用线程) =====

void ControlNode::ControlLoop()
{
  auto next_cycle = std::chrono::steady_clock::now();

  while (running_)
  {
    // 精确周期控制：sleep_until 无累积误差
    next_cycle += cycle_period_;
    std::this_thread::sleep_until(next_cycle);

    // 检查消息是否就绪
    if (!traj_received_ || !loc_received_ || !chassis_received_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Waiting for messages: traj=%d loc=%d chassis=%d",
                           traj_received_.load(), loc_received_.load(), chassis_received_.load());
      continue;
    }

    // 读取缓存 (shared_ptr, 线程安全)
    auto traj = latest_trajectory_;
    auto loc = latest_localization_;
    auto chs = latest_chassis_;

    std::vector<TrajectoryPoint> traj_points = ConvertTrajectory(traj);
    StateVec current_state = GetCurrentState();

    if (traj_points.empty()) continue;

    int closest_idx = FindClosestPointIndex(current_state(0), current_state(1), traj_points);
    const auto & ref_pt = traj_points[closest_idx];

    double lateral_err = ComputeLateralError(current_state(0), current_state(1),
                                             current_state(2), ref_pt);
    double heading_err = ComputeHeadingError(current_state(2), ref_pt);
    double velocity_err = current_state(3) - ref_pt.velocity;

    // 异常检测
    double now_sec = this->now().seconds();
    AnomalyStatus anomaly = anomaly_detector_->Detect(
        rclcpp::Time(traj->header.stamp).seconds(),
        rclcpp::Time(loc->header.stamp).seconds(),
        rclcpp::Time(chs->header.stamp).seconds(),
        now_sec, traj_points,
        current_state(0), current_state(1), current_state(2),
        lateral_err, heading_err);

    // 构建 debug 消息
    auto debug_msg = msg::DebugInfo();
    debug_msg.header.stamp = this->now();
    debug_msg.lateral_error = lateral_err;
    debug_msg.heading_error = heading_err;
    debug_msg.velocity_error = velocity_err;
    debug_msg.status = anomaly.is_ok ? "OK" : "ANOMALY";
    for (const auto & w : anomaly.warnings) debug_msg.warnings.push_back(w);

    if (!anomaly.is_ok) {
      debug_msg.mpc_solve_time_ms = 0.0;
      debug_pub_->publish(debug_msg);
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "Anomaly: %zu warnings", anomaly.warnings.size());
      continue;
    }

    // MPC 求解
    MPCController::ControlResult result = mpc_controller_->Solve(current_state, traj_points);

    // 发布控制指令
    auto ctrl_msg = msg::ControlCommand();
    ctrl_msg.header.stamp = this->now();
    ctrl_msg.steering_angle_target = result.steering_angle;
    ctrl_msg.acceleration_target = result.acceleration;
    control_pub_->publish(ctrl_msg);

    // 发布 debug
    debug_msg.mpc_solve_time_ms = result.solve_time_ms;
    debug_msg.predicted_trajectory_x = result.predicted_x;
    debug_msg.predicted_trajectory_y = result.predicted_y;
    debug_pub_->publish(debug_msg);
  }
}

}  // namespace control_pkg

// 主函数: 主线程只做 IO (spin)
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<control_pkg::ControlNode>();
  rclcpp::spin(node);  // 主线程 = IO 线程, 只跑订阅回调
  rclcpp::shutdown();
  return 0;
}