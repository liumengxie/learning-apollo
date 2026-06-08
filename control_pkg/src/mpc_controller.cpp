#include "control_pkg/mpc_controller.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>

namespace control_pkg
{

MPCController::MPCController(const Config & config)
  : config_(config), vehicle_model_(config.wheelbase), last_control_(ControlVec::Zero()) {}

std::vector<StateVec> MPCController::GetReferenceTrajectory(
    const StateVec & current_state,
    const std::vector<TrajectoryPoint> & trajectory) const
{
  std::vector<StateVec> ref_states;
  ref_states.reserve(config_.prediction_horizon);
  if (trajectory.empty()) return ref_states;

  // 找最近点
  int closest_idx = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < trajectory.size(); ++i) {
    double dx = trajectory[i].x - current_state(0);
    double dy = trajectory[i].y - current_state(1);
    double dist = dx * dx + dy * dy;
    if (dist < min_dist) { min_dist = dist; closest_idx = static_cast<int>(i); }
  }

  // 沿轨迹取 N 个参考点
  int prev_idx = closest_idx;
  for (int k = 0; k < config_.prediction_horizon; ++k) {
    double target_dist = k * config_.dt * std::max(current_state(3), 1.0);
    int idx = closest_idx;
    double accumulated_dist = 0.0;
    prev_idx = closest_idx;

    for (size_t i = closest_idx + 1; i < trajectory.size(); ++i) {
      double dx = trajectory[i].x - trajectory[prev_idx].x;
      double dy = trajectory[i].y - trajectory[prev_idx].y;
      accumulated_dist += std::sqrt(dx * dx + dy * dy);
      if (accumulated_dist >= target_dist) { idx = static_cast<int>(i); break; }
      prev_idx = static_cast<int>(i); idx = prev_idx;
    }
    idx = std::min(idx, static_cast<int>(trajectory.size()) - 1);

    StateVec ref;
    ref << trajectory[idx].x, trajectory[idx].y,
           trajectory[idx].yaw, trajectory[idx].velocity;
    ref_states.push_back(ref);
  }
  return ref_states;
}

void MPCController::BuildQPProblem(const StateVec & current_state,
                                   const std::vector<StateVec> & ref_trajectory,
                                   Eigen::MatrixXd & H, Eigen::VectorXd & g) const
{
  const int N = config_.prediction_horizon, nx = 4, nu = 2, dim_u = N * nu;

  H = Eigen::MatrixXd::Zero(dim_u, dim_u);
  g = Eigen::VectorXd::Zero(dim_u);

  // 逐步线性化
  StateVec x = current_state;
  ControlVec u_op = ControlVec::Zero();
  std::vector<Eigen::Matrix4d> A_list(N);
  std::vector<Eigen::Matrix<double, 4, 2>> B_list(N);
  for (int k = 0; k < N; ++k) {
    vehicle_model_.Linearize(x, u_op, config_.dt, A_list[k], B_list[k]);
    x = vehicle_model_.Update(x, u_op, config_.dt);
  }

  // B_qp 矩阵
  Eigen::MatrixXd B_qp = Eigen::MatrixXd::Zero(N * nx, N * nu);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      Eigen::Matrix<double, 4, 2> Bij = B_list[j];
      Eigen::Matrix4d A_prod = Eigen::Matrix4d::Identity();
      for (int k = j + 1; k <= i; ++k) A_prod = A_list[k] * A_prod;
      B_qp.block(i * nx, j * nu, nx, nu) = A_prod * Bij;
    }
  }

  // Q_bar, R_bar, Rd_bar
  Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(N * nx, N * nx);
  Eigen::MatrixXd R_bar = Eigen::MatrixXd::Zero(N * nu, N * nu);
  for (int k = 0; k < N; ++k) {
    Q_bar.block(k * nx, k * nx, nx, nx) = config_.Q;
    R_bar.block(k * nu, k * nu, nu, nu) = config_.R;
  }

  // 变化率惩罚矩阵 M
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(N * nu, N * nu);
  for (int k = 1; k < N; ++k)
    M.block(k * nu, (k - 1) * nu, nu, nu) = -Eigen::Matrix2d::Identity();
  Eigen::MatrixXd Rd_bar = Eigen::MatrixXd::Zero(N * nu, N * nu);
  for (int k = 0; k < N; ++k)
    Rd_bar.block(k * nu, k * nu, nu, nu) = config_.Rd;

  // H = B'Q B + R + M'Rd M
  H = B_qp.transpose() * Q_bar * B_qp + R_bar + M.transpose() * Rd_bar * M;
  H = 0.5 * (H + H.transpose());
  for (int i = 0; i < H.rows(); ++i) H(i, i) += 1e-6;

  // g = B'Q (A x0 - Xref) + 变化率初始项
  Eigen::VectorXd X_ref(N * nx);
  for (int k = 0; k < N; ++k) X_ref.segment(k * nx, nx) = ref_trajectory[k];

  // A_qp * x0
  Eigen::VectorXd A_qp_x0 = Eigen::VectorXd::Zero(N * nx);
  Eigen::Matrix4d A_pow = Eigen::Matrix4d::Identity();
  for (int i = 0; i < N; ++i) {
    A_pow = A_list[i] * A_pow;
    A_qp_x0.segment(i * nx, nx) = A_pow * current_state;
  }

  g = B_qp.transpose() * Q_bar * (A_qp_x0 - X_ref);

  Eigen::VectorXd u_prev_aug = Eigen::VectorXd::Zero(N * nu);
  u_prev_aug.head(nu) = last_control_;
  g += M.transpose() * Rd_bar * M * (-u_prev_aug);
}

Eigen::VectorXd MPCController::SolveQP(const Eigen::MatrixXd & H,
                                       const Eigen::VectorXd & g) const
{
  Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
  return ldlt.solve(-g);
}

void MPCController::ApplyConstraints(Eigen::VectorXd & U,
                                     const ControlVec & last_control) const
{
  const int N = config_.prediction_horizon, nu = 2;
  const double max_steer = config_.max_steering_angle;
  const double max_accel = config_.max_acceleration;
  const double min_accel = config_.max_deceleration;
  const double max_steer_rate = config_.max_steering_rate * config_.dt;
  const double max_accel_rate = (max_accel - min_accel) * config_.dt;

  for (int k = 0; k < N; ++k) {
    int idx = k * nu;
    U(idx) = std::clamp(U(idx), -max_steer, max_steer);
    U(idx + 1) = std::clamp(U(idx + 1), min_accel, max_accel);

    double prev_steer = (k == 0) ? last_control(0) : U((k - 1) * nu);
    double prev_accel = (k == 0) ? last_control(1) : U((k - 1) * nu + 1);

    U(idx) = prev_steer + std::clamp(U(idx) - prev_steer, -max_steer_rate, max_steer_rate);
    U(idx + 1) = prev_accel + std::clamp(U(idx + 1) - prev_accel, -max_accel_rate, max_accel_rate);
  }
}

void MPCController::GetControlFromSolution(const Eigen::VectorXd & U,
                                           double & steering, double & accel) const
{
  steering = U(0); accel = U(1);
}

void MPCController::PredictTrajectory(const StateVec & current_state,
                                      const Eigen::VectorXd & U,
                                      std::vector<double> & pred_x,
                                      std::vector<double> & pred_y) const
{
  pred_x.clear(); pred_y.clear();
  const int N = config_.prediction_horizon, nu = 2;
  StateVec state = current_state;
  pred_x.push_back(state(0)); pred_y.push_back(state(1));
  for (int k = 0; k < N; ++k) {
    ControlVec ctrl; ctrl << U(k * nu), U(k * nu + 1);
    state = vehicle_model_.Update(state, ctrl, config_.dt);
    pred_x.push_back(state(0)); pred_y.push_back(state(1));
  }
}

MPCController::ControlResult MPCController::Solve(
    const StateVec & current_state,
    const std::vector<TrajectoryPoint> & reference_trajectory)
{
  ControlResult result;
  result.success = false;

  auto t_start = std::chrono::high_resolution_clock::now();

  std::vector<StateVec> ref_traj = GetReferenceTrajectory(current_state, reference_trajectory);
  if (ref_traj.empty()) return result;

  Eigen::MatrixXd H; Eigen::VectorXd g;
  BuildQPProblem(current_state, ref_traj, H, g);

  Eigen::VectorXd U = SolveQP(H, g);
  ApplyConstraints(U, last_control_);

  GetControlFromSolution(U, result.steering_angle, result.acceleration);
  PredictTrajectory(current_state, U, result.predicted_x, result.predicted_y);

  last_control_ << result.steering_angle, result.acceleration;

  auto t_end = std::chrono::high_resolution_clock::now();
  result.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  result.success = true;
  return result;
}

}  // namespace control_pkg