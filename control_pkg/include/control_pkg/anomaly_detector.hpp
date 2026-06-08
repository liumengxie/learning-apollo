#ifndef CONTROL_PKG__ANOMALY_DETECTOR_HPP_
#define CONTROL_PKG__ANOMALY_DETECTOR_HPP_

#include "control_pkg/types.hpp"
#include <vector>

namespace control_pkg
{

class AnomalyDetector
{
public:
  AnomalyDetector(double max_msg_age_ms = 500.0,
                  double max_lateral_error = 2.0,
                  double max_heading_error = 1.0);

  bool CheckMessageTimeout(double last_msg_time_sec, double current_time_sec) const;
  bool CheckLocalizationValid(double x, double y, double yaw) const;
  bool CheckTrajectoryValid(const std::vector<TrajectoryPoint> & points) const;
  bool CheckTrackingError(double lateral_error, double heading_error) const;

  AnomalyStatus Detect(double traj_time_sec, double loc_time_sec,
                       double chassis_time_sec, double current_time_sec,
                       const std::vector<TrajectoryPoint> & traj,
                       double x, double y, double yaw,
                       double lateral_error, double heading_error) const;

private:
  double max_msg_age_ms_, max_lateral_error_, max_heading_error_;
};

}  // namespace control_pkg

#endif