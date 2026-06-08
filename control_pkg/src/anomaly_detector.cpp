#include "control_pkg/anomaly_detector.hpp"
#include <cmath>

namespace control_pkg
{

AnomalyDetector::AnomalyDetector(double max_msg_age_ms,
                                 double max_lateral_error,
                                 double max_heading_error)
  : max_msg_age_ms_(max_msg_age_ms),
    max_lateral_error_(max_lateral_error),
    max_heading_error_(max_heading_error) {}

bool AnomalyDetector::CheckMessageTimeout(double last_msg_time_sec,
                                          double current_time_sec) const
{
  return (current_time_sec - last_msg_time_sec) * 1000.0 <= max_msg_age_ms_;
}

bool AnomalyDetector::CheckLocalizationValid(double x, double y, double yaw) const
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw)) return false;
  if (yaw < -M_PI - 1e-3 || yaw > M_PI + 1e-3) return false;
  return true;
}

bool AnomalyDetector::CheckTrajectoryValid(
    const std::vector<TrajectoryPoint> & points) const
{
  if (points.size() < 2) return false;
  for (const auto & pt : points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
        !std::isfinite(pt.yaw) || !std::isfinite(pt.velocity)) return false;
  }
  return true;
}

bool AnomalyDetector::CheckTrackingError(double lateral_error,
                                         double heading_error) const
{
  if (!std::isfinite(lateral_error) || !std::isfinite(heading_error)) return false;
  if (std::abs(lateral_error) > max_lateral_error_) return false;
  if (std::abs(heading_error) > max_heading_error_) return false;
  return true;
}

AnomalyStatus AnomalyDetector::Detect(
    double traj_time_sec, double loc_time_sec,
    double chassis_time_sec, double current_time_sec,
    const std::vector<TrajectoryPoint> & traj,
    double x, double y, double yaw,
    double lateral_error, double heading_error) const
{
  AnomalyStatus status;
  status.is_ok = true;

  if (!CheckMessageTimeout(traj_time_sec, current_time_sec)) {
    status.is_ok = false;
    status.warnings.push_back("Trajectory message timeout");
  }
  if (!CheckMessageTimeout(loc_time_sec, current_time_sec)) {
    status.is_ok = false;
    status.warnings.push_back("Localization message timeout");
  }
  if (!CheckMessageTimeout(chassis_time_sec, current_time_sec)) {
    status.is_ok = false;
    status.warnings.push_back("Chassis message timeout");
  }
  if (!CheckLocalizationValid(x, y, yaw)) {
    status.is_ok = false;
    status.warnings.push_back("Localization data invalid");
  }
  if (!CheckTrajectoryValid(traj)) {
    status.is_ok = false;
    status.warnings.push_back("Trajectory data invalid");
  }
  if (!CheckTrackingError(lateral_error, heading_error)) {
    status.is_ok = false;
    status.warnings.push_back("Tracking error exceeds threshold");
  }
  return status;
}

}  // namespace control_pkg