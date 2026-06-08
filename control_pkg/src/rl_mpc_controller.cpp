#include "control_pkg/rl_mpc_controller.hpp"
#include <cmath>
#include <chrono>
#include <algorithm>

namespace control_pkg
{

RLMPCController::RLMPCController(const Config & config)
  : config_(config), mpc_(config.mpc_config), value_network_(config.vf_config),
    buffer_(config.replay_buffer_capacity)
{
  last_state_ = Eigen::Vector4d::Zero();
  last_action_ = Eigen::Vector2d::Zero();
}

double RLMPCController::ComputeReward(double lat_err, double head_err, double vel_err) const
{
  return -(config_.reward_w_lat * lat_err * lat_err
         + config_.reward_w_yaw * head_err * head_err
         + config_.reward_w_vel * vel_err * vel_err);
}

void RLMPCController::StoreTransition(const Eigen::Vector4d & state,
                                      const Eigen::Vector2d & action,
                                      double reward,
                                      const Eigen::Vector4d & next_state,
                                      bool done)
{
  ReplayBuffer::Transition tr;
  tr.state = state;
  tr.action = action;
  tr.reward = reward;
  tr.next_state = next_state;
  tr.done = done;
  buffer_.Push(tr);
}

void RLMPCController::IntegrateValueFunction(
    const StateVec & current_state,
    const std::vector<StateVec> & ref_trajectory,
    Eigen::MatrixXd & H, Eigen::VectorXd & g) const
{
  if (ref_trajectory.empty()) return;

  const int N = config_.mpc_config.prediction_horizon;
  const int nx = 4, nu = 2, dim_u = N * nu;

  // 获取终端参考状态 x_ref_N
  const auto & x_ref_N = ref_trajectory.back();

  // 在参考终点查询 Value Network 的梯度和 Hessian
  // 注意: 需要 lat/head/vel error, 但这里在参考点上误差=0
  Eigen::Matrix4d V_hess = value_network_.GetHessian(
      Eigen::Vector4d(x_ref_N(0), x_ref_N(1), x_ref_N(2), x_ref_N(3)),
      0.0, 0.0, 0.0);

  // 保证 Hessian 正定
  for (int i = 0; i < 4; ++i)
    V_hess(i, i) = std::max(V_hess(i, i), config_.hessian_reg);

  // 检查是否正定, 不是则替换为单位矩阵
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(V_hess);
  if (eigensolver.eigenvalues()(0) <= 0) {
    V_hess = Eigen::Matrix4d::Identity() * config_.hessian_reg;
  }

  // 计算梯度
  Eigen::Matrix<double, 7, 1> s7;
  s7 << x_ref_N(0), x_ref_N(1), x_ref_N(2), x_ref_N(3), 0, 0, 0;
  Eigen::VectorXd V_grad_full = value_network_.GetGradient(s7);
  Eigen::Vector4d V_grad = V_grad_full.head<4>();

  // ---- 修改 QP 的 H 和 g ----
  // 终端代价 = V(x_N) ≈ V(x_ref_N) + g'(x_N - x_ref_N) + ½(x_N - x_ref_N)'H(x_N - x_ref_N)
  // → Q_f = H (终端状态权重) + V_hess
  // → 参考终点修正: X_ref_Nloc -= Q_f^{-1} * V_grad

  // Q_f = 原Q + V_hess
  Eigen::Matrix4d Q_f = config_.mpc_config.Q + V_hess * 0.01;  // scale down V_hess contribution

  // 修改 H 矩阵的终端块
  // H 矩阵对应 U = [u_0; u_1; ...; u_{N-1}]
  // 终端代价通过 B_qp 和 A_qp 间接映射到 H
  // 简化: 直接修改 H 的最后 nu 块 (对应最后一个控制量的影响)
  // 完整实现需要重新构建带终端代价的 QP, 这里采用近似:

  // 在 g 中添加终端梯度项: g += B_qp' * Q_bar * [0;...;0; -Q^{-1} g_V]
  // 这由 BuildQPProblem 内部完成, 这里只是补充修改
  // 简化: 直接修改 X_ref 中的终端参考
  // 实际的 X_ref_N 修正通过 g 完成, 即 g += (与V_grad相关项)

  // 由于修改 H 和 g 会大幅增加复杂度, 这里用更简单的方式:
  // 将 V_hess 用作 Q 的终端放大, 将 V_grad 作为一个额外的终端状态偏差
  Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(N * nx, N * nx);
  for (int k = 0; k < N; ++k) {
    if (k == N - 1) {
      Q_bar.block(k * nx, k * nx, nx, nx) = Q_f;  // 终端用增强版
    } else {
      Q_bar.block(k * nx, k * nx, nx, nx) = config_.mpc_config.Q;
    }
  }

  // 终端参考修正: X_ref(N-1) += step * Q_f^{-1} * (-V_grad)
  // 但这里不修改 ref_trajectory (const引用), 所以修改发送给外部QP的数据
  // 此函数暂不修改H和g, 完整逻辑在 Solve() 中调用后处理
  // (见 RLMPCController::Solve)

  (void)H; (void)g; (void)dim_u; (void)current_state;
}

MPCController::ControlResult RLMPCController::Solve(
    const StateVec & current_state,
    const std::vector<TrajectoryPoint> & ref_traj,
    double lat_err, double head_err, double vel_err)
{
  // Step 1: 标准 MPC 求解
  MPCController::ControlResult result = mpc_.Solve(current_state, ref_traj);
  if (!result.success) return result;

  // Step 2: 用 Value Function 的终端梯度修正控制量
  // V_grad 指示终端价值变化方向, 调整控制以偏向更高价值区域
  std::vector<StateVec> ref_states = mpc_.GetReferenceTrajectory(current_state, ref_traj);
  if (!ref_states.empty())
  {
    const auto & x_ref_last = ref_states.back();

    Eigen::Matrix<double, 7, 1> s7;
    s7 << x_ref_last(0), x_ref_last(1), x_ref_last(2), x_ref_last(3), lat_err, head_err, vel_err;
    Eigen::VectorXd V_grad_full = value_network_.GetGradient(s7);
    Eigen::Vector4d V_grad = V_grad_full.head<4>();

    // 梯度方向的控制修正
    // 如果 ∂V/∂yaw > 0 (更高的价值在正航向方向), 则增加转向
    // 如果 ∂V/∂v > 0 (更高的价值在加速方向), 则增加加速度
    double steer_correction = 0.05 * (V_grad(2)) / (std::abs(V_grad(2)) + 1.0);
    double accel_correction = 0.1 * (V_grad(3)) / (std::abs(V_grad(3)) + 1.0);

    // 限幅
    steer_correction = std::clamp(steer_correction, -0.05, 0.05);
    accel_correction = std::clamp(accel_correction, -0.3, 0.3);

    result.steering_angle += steer_correction;
    result.acceleration += accel_correction;
  }

  // Step 3: 存储 transition
  Eigen::Vector2d action(result.steering_angle, result.acceleration);
  double reward = ComputeReward(lat_err, head_err, vel_err);

  if (step_count_ > 0) {
    StoreTransition(last_state_, last_action_, reward, current_state, false);
  }

  last_state_ = current_state;
  last_action_ = action;

  step_count_++;

  return result;
}

void RLMPCController::Reset()
{
  mpc_.Reset();
  buffer_.Clear();
  step_count_ = 0;
  train_count_ = 0;
  last_state_ = Eigen::Vector4d::Zero();
  last_action_ = Eigen::Vector2d::Zero();
}

}  // namespace control_pkg