#ifndef CONTROL_PKG__VEHICLE_MODEL_HPP_
#define CONTROL_PKG__VEHICLE_MODEL_HPP_

#include "control_pkg/types.hpp"
#include <Eigen/Dense>

namespace control_pkg
{

/// @brief 运动学自行车模型
/// 状态: [x, y, yaw, v], 控制: [steering_angle, acceleration]
class VehicleModel
{
public:
  explicit VehicleModel(double wheelbase);

  StateVec Update(const StateVec & state, const ControlVec & control, double dt) const;

  void Linearize(const StateVec & state, const ControlVec & control, double dt,
                 Eigen::Matrix4d & A, Eigen::Matrix<double, 4, 2> & B) const;

  double GetWheelbase() const { return wheelbase_; }

private:
  double wheelbase_;
};

}  // namespace control_pkg

#endif