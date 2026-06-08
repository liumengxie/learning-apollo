#include "control_pkg/vehicle_model.hpp"
#include <cmath>

namespace control_pkg
{

VehicleModel::VehicleModel(double wheelbase) : wheelbase_(wheelbase) {}

StateVec VehicleModel::Update(const StateVec & state, const ControlVec & control,
                              double dt) const
{
  const double x = state(0), y = state(1), yaw = state(2), v = state(3);
  const double steer = control(0), accel = control(1);

  StateVec next;
  next(0) = x + v * std::cos(yaw) * dt;
  next(1) = y + v * std::sin(yaw) * dt;
  next(2) = yaw + v / wheelbase_ * std::tan(steer) * dt;
  next(3) = v + accel * dt;
  return next;
}

void VehicleModel::Linearize(const StateVec & state, const ControlVec & control,
                             double dt, Eigen::Matrix4d & A,
                             Eigen::Matrix<double, 4, 2> & B) const
{
  const double yaw = state(2), v = state(3), steer = control(0);

  A.setIdentity();
  A(0, 2) = -v * std::sin(yaw) * dt;
  A(0, 3) = std::cos(yaw) * dt;
  A(1, 2) = v * std::cos(yaw) * dt;
  A(1, 3) = std::sin(yaw) * dt;
  A(2, 3) = std::tan(steer) / wheelbase_ * dt;

  B.setZero();
  B(2, 0) = v / wheelbase_ * (1.0 + std::tan(steer) * std::tan(steer)) * dt;
  B(3, 1) = dt;
}

}  // namespace control_pkg