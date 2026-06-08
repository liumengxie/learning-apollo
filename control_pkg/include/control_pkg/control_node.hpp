#ifndef CONTROL_PKG__CONTROL_NODE_HPP_
#define CONTROL_PKG__CONTROL_NODE_HPP_

#include "control_pkg/types.hpp"
#include "control_pkg/mpc_controller.hpp"
#include "control_pkg/anomaly_detector.hpp"
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <atomic>
#include <chrono>

namespace control_pkg
{
namespace msg
{
struct Trajectory;
struct TrajectoryPoint;
struct Localization;
struct Chassis;
struct ControlCommand;
struct DebugInfo;
}
}

namespace control_pkg
{

class ControlNode : public rclcpp::Node
{
public:
  explicit ControlNode(const std::string & node_name = "control_node");
  ~ControlNode();

private:
  // 订阅回调 (仅做缓存, ~1μs)
  void TrajectoryCallback(const std::shared_ptr<msg::Trajectory> msg);
  void LocalizationCallback(const std::shared_ptr<msg::Localization> msg);
  void ChassisCallback(const std::shared_ptr<msg::Chassis> msg);

  // 控制循环 (专用线程, while+sleep_until 精确周期)
  void ControlLoop();

  // 工具
  double ComputeLateralError(double x, double y, double heading,
                             const TrajectoryPoint & ref_point) const;
  double ComputeHeadingError(double heading, const TrajectoryPoint & ref_point) const;
  int FindClosestPointIndex(double x, double y,
                            const std::vector<TrajectoryPoint> & traj) const;
  StateVec GetCurrentState() const;
  std::vector<TrajectoryPoint> ConvertTrajectory(
      const std::shared_ptr<msg::Trajectory> & msg) const;

  // 发布者
  rclcpp::Publisher<msg::ControlCommand>::SharedPtr control_pub_;
  rclcpp::Publisher<msg::DebugInfo>::SharedPtr debug_pub_;

  // 订阅者
  rclcpp::Subscription<msg::Trajectory>::SharedPtr traj_sub_;
  rclcpp::Subscription<msg::Localization>::SharedPtr loc_sub_;
  rclcpp::Subscription<msg::Chassis>::SharedPtr chassis_sub_;

  // 核心组件
  std::unique_ptr<MPCController> mpc_controller_;
  std::unique_ptr<AnomalyDetector> anomaly_detector_;

  // 数据缓存 (shared_ptr原子引用计数, 天然线程安全)
  std::shared_ptr<msg::Trajectory> latest_trajectory_;
  std::shared_ptr<msg::Localization> latest_localization_;
  std::shared_ptr<msg::Chassis> latest_chassis_;
  std::atomic<bool> traj_received_{false};
  std::atomic<bool> loc_received_{false};
  std::atomic<bool> chassis_received_{false};

  // 控制线程
  std::thread control_thread_;
  std::atomic<bool> running_{true};
  std::chrono::nanoseconds cycle_period_;

  double control_frequency_{50.0};
};

}  // namespace control_pkg

#endif