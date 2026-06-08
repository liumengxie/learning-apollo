#ifndef CONTROL_PKG__RL_MPC_CONTROLLER_HPP_
#define CONTROL_PKG__RL_MPC_CONTROLLER_HPP_

#include "control_pkg/mpc_controller.hpp"
#include "control_pkg/value_network.hpp"
#include "control_pkg/replay_buffer.hpp"
#include "control_pkg/types.hpp"

namespace control_pkg
{

/// @brief RL增强MPC控制器
///
/// 在标准MPC的代价函数中用学习的 Value Function Vθ(x_N) 替换终端代价:
///   J = Σ(x_k - x_ref)'Q(x_k - x_ref) + u_k'R u_k  +  Vθ(x_N)
///
/// Vθ 在轨迹终点做二阶泰勒展开, 集成到 QP:
///   - 终端Q矩阵: Q_f += ∇²Vθ(x_ref_N)  (Hessian)
///   - 终端参考偏移: x_ref_N -= Q_f^{-1} * ∇Vθ(x_ref_N)  (梯度)
///
/// 训练: 每个控制周期存储 transition, 每10步执行一次 TD-Learning
class RLMPCController
{
public:
  struct Config
  {
    MPCController::Config mpc_config;
    ValueNetwork::Config vf_config;
    size_t replay_buffer_capacity = 100000;
    int train_interval = 10;       // 每 N 个控制周期训练一次
    int train_batch_size = 64;     // 每次训练的 batch 大小
    double reward_w_lat = 1.0;     // 横向误差权重
    double reward_w_yaw = 5.0;     // 航向误差权重
    double reward_w_vel = 0.5;     // 速度误差权重
    double hessian_reg = 1e-4;     // Hessian 正则化 (保证正定)
  };

  /// @param config RL-MPC 配置
  explicit RLMPCController(const Config & config);

  /// @brief 主求解入口 (替代 MPCController::Solve)
  /// @param state 当前状态 [x, y, yaw, v]
  /// @param ref_traj 参考轨迹
  /// @param lat_err 横向误差
  /// @param head_err 航向误差
  /// @param vel_err 速度误差
  MPCController::ControlResult Solve(const StateVec & state,
                                     const std::vector<TrajectoryPoint> & ref_traj,
                                     double lat_err, double head_err, double vel_err);

  /// @brief 存储 transition 到 replay buffer
  /// @param state, action, reward, next_state, done
  void StoreTransition(const Eigen::Vector4d & state, const Eigen::Vector2d & action,
                       double reward, const Eigen::Vector4d & next_state, bool done);

  /// @brief 训练一步 (TD-Learning)
  /// @return 平均 loss
  double Train();

  /// @brief 是否需要训练
  bool ShouldTrain() const { return (step_count_ % config_.train_interval == 0) && buffer_.Size() >= static_cast<size_t>(config_.train_batch_size); }

  /// @brief 计算奖励
  double ComputeReward(double lat_err, double head_err, double vel_err) const;

  /// @brief 获取 Value Network
  ValueNetwork & GetValueNetwork() { return value_network_; }
  const ValueNetwork & GetValueNetwork() const { return value_network_; }

  /// @brief 底层 MPC 控制器
  MPCController & GetMPC() { return mpc_; }

  /// @brief 步数计数
  int StepCount() const { return step_count_; }

  /// @brief 训练计数
  int TrainCount() const { return train_count_; }

  /// @brief 重置
  void Reset();

private:
  /// @brief 修改 QP 问题的终端代价矩阵 (集成 Vθ 信息)
  void IntegrateValueFunction(const StateVec & current_state,
                              const std::vector<StateVec> & ref_trajectory,
                              Eigen::MatrixXd & H, Eigen::VectorXd & g) const;

  Config config_;
  MPCController mpc_;
  ValueNetwork value_network_;
  ReplayBuffer buffer_;

  // 缓存的上一帧数据
  Eigen::Vector4d last_state_;
  Eigen::Vector2d last_action_;

  int step_count_{0};
  int train_count_{0};
};

}  // namespace control_pkg

#endif