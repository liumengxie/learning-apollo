#ifndef CONTROL_PKG__VALUE_NETWORK_HPP_
#define CONTROL_PKG__VALUE_NETWORK_HPP_

#include "control_pkg/neural_network.hpp"
#include "control_pkg/replay_buffer.hpp"
#include <Eigen/Dense>

namespace control_pkg
{

/// @brief 基于TD-Learning的Value function Vθ(s)
///
/// 使用双网络架构:
///   - online network:  用于训练和推理
///   - target network:  用于计算TD target (软更新)
///
/// 奖励定义: r = -(lat_err² + w_yaw * heading_err² + w_vel * vel_err²)
///
/// 输入: 7维 [x, y, yaw, v, lat_err, heading_err, vel_err]
/// 输出: 1维标量 V(s)
class ValueNetwork
{
public:
  struct Config
  {
    int input_dim;
    int hidden_dim;
    double learning_rate;
    double gamma;
    double tau;
    double weight_decay;
    double grad_clip;

    Config()
      : input_dim(7), hidden_dim(64), learning_rate(1e-3),
        gamma(0.99), tau(0.005), weight_decay(1e-5), grad_clip(1.0) {}
  };

  explicit ValueNetwork(const Config & config = Config());

  // ---- 推理 ----

  /// @brief 估计状态价值 V(s)
  double Predict(const Eigen::VectorXd & state) const;

  /// @brief 获取 V(s) 在 state 处的梯度 ∇V
  Eigen::VectorXd GetGradient(const Eigen::VectorXd & state) const;

  /// @brief 获取 V(s) 在 state 处的 Hessian ∇²V
  /// @param state 4维状态 [x, y, yaw, v]
  /// @param lat_err  横向误差
  /// @param head_err 航向误差
  /// @param vel_err  速度误差
  Eigen::Matrix4d GetHessian(const Eigen::Vector4d & state,
                             double lat_err, double head_err, double vel_err) const;

  // ---- 训练 ----

  /// @brief 执行一步 TD-Learning 更新
  /// @param batch 采样的一批 transitions
  /// @return 平均 loss
  double TrainStep(const std::vector<ReplayBuffer::Transition> & batch);

  /// @brief 软更新 target network
  void UpdateTarget();

  // ---- 保存/加载 ----
  void SaveWeights(const std::string & filepath) const;
  void LoadWeights(const std::string & filepath);

  const NeuralNetwork & OnlineNetwork() const { return online_; }
  const Config & GetConfig() const { return config_; }

private:
  /// @brief 构建7维增强状态
  Eigen::Matrix<double, 7, 1> BuildState(const Eigen::Vector4d & state,
                                         double lat_err, double head_err,
                                         double vel_err) const;

  Config config_;
  NeuralNetwork online_;
  NeuralNetwork target_;

  // Adam 优化器状态
  double beta1_{0.9}, beta2_{0.999}, epsilon_{1e-8};
  int train_step_count_{0};
  NeuralNetwork::Gradients m_;   // 一阶动量
  NeuralNetwork::Gradients v_;   // 二阶动量
};

}  // namespace control_pkg

#endif