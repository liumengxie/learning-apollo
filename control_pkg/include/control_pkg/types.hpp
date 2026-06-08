#ifndef CONTROL_PKG__TYPES_HPP_
#define CONTROL_PKG__TYPES_HPP_

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace control_pkg
{

struct TrajectoryPoint
{
  double x, y, z, yaw, velocity, curvature, relative_time;
};

struct AnomalyStatus
{
  bool is_ok;
  std::vector<std::string> warnings;
};

using StateVec = Eigen::Vector4d;   // [x, y, yaw, v]
using ControlVec = Eigen::Vector2d; // [steering_angle, acceleration]

}  // namespace control_pkg

#endif