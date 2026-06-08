#ifndef CONTROL_PKG__MPC_CONTROLLER_HPP_
#define CONTROL_PKG__MPC_CONTROLLER_HPP_

#include "control_pkg/types.hpp"
#include "control_pkg/vehicle_model.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace control_pkg
{

class MPCController
{
public:
  struct Config
  {
    int prediction_horizon = 20;
    double dt = 0.05;
    double wheelbase = 2.8;
    double max_steering_angle = 0.6;
    double max_steering_rate = 0.5;
    double max_acceleration = 3.0;
    double max_deceleration = -5.0;
    Eigen::Matrix4d Q;
    Eigen::Matrix2d R;
    Eigen::Matrix2d Rd;

    Config()
    {
      Q.setIdentity(); Q.diagonal() << 1.0, 1.0, 5.0, 0.5;
      R.setIdentity(); R.diagonal() << 10.0, 1.0;
      Rd.setIdentity(); Rd.diagonal() << 20.0, 2.0;
    }
  };

  struct ControlResult
  {
    double steering_angle, acceleration, solve_time_ms;
    std::vector<double> predicted_x, predicted_y;
    bool success;
  };

  explicit MPCController(const Config & config);

  ControlResult Solve(const StateVec & current_state,
                      const std::vector<TrajectoryPoint> & reference_trajectory);

  std::vector<StateVec> GetReferenceTrajectory(
      const StateVec & current_state,
      const std::vector<TrajectoryPoint> & trajectory) const;

  void BuildQPProblem(const StateVec & current_state,
                      const std::vector<StateVec> & ref_trajectory,
                      Eigen::MatrixXd & H, Eigen::VectorXd & g) const;

  Eigen::VectorXd SolveQP(const Eigen::MatrixXd & H, const Eigen::VectorXd & g) const;

  void ApplyConstraints(Eigen::VectorXd & U, const ControlVec & last_control) const;

  void GetControlFromSolution(const Eigen::VectorXd & U,
                              double & steering, double & accel) const;

  void PredictTrajectory(const StateVec & current_state, const Eigen::VectorXd & U,
                         std::vector<double> & pred_x, std::vector<double> & pred_y) const;

  void Reset() { last_control_ = ControlVec::Zero(); }

private:
  Config config_;
  VehicleModel vehicle_model_;
  ControlVec last_control_;
};

}  // namespace control_pkg

#endif