#ifndef CONTROL_PKG__NEURAL_NETWORK_HPP_
#define CONTROL_PKG__NEURAL_NETWORK_HPP_

#include <Eigen/Dense>
#include <vector>
#include <random>

namespace control_pkg
{

/// @brief 3层全连接神经网络 (ReLU + ReLU + Linear)
///
/// 结构: input → hidden → hidden → output
/// 权重使用 Xavier 初始化, bias = 0
///
/// 提供:
///   - Forward: y = W3 σ(W2 σ(W1 x + b1) + b2) + b3
///   - Backward: MSE loss 梯度, 返回各层 dW / db
///   - Gradient: ∂y/∂x (自动微分)
///   - Hessian:  ∂²y/∂x² (数值差分)
class NeuralNetwork
{
public:
  struct Weights
  {
    Eigen::MatrixXd W1, W2, W3;
    Eigen::VectorXd b1, b2, b3;
  };

  struct Gradients
  {
    Eigen::MatrixXd dW1, dW2, dW3;
    Eigen::VectorXd db1, db2, db3;
  };

  /// @brief 构造 3 层网络
  /// @param input_dim  输入维度 (状态空间 = 4)
  /// @param hidden_dim 隐藏层维度
  /// @param output_dim 输出维度 (= 1)
  NeuralNetwork(int input_dim, int hidden_dim, int output_dim);

  /// @brief 前向传播
  /// @param x 输入向量
  /// @return 输出标量
  double Forward(const Eigen::VectorXd & x) const;

  /// @brief 前向传播, 同时返回各层激活值 (用于反向传播和梯度计算)
  /// @param x 输入
  /// @param cache_h1 输出: W1*x+b1 (激活前)
  /// @param cache_a1 输出: σ(z1)
  /// @param cache_h2 输出: W2*a1+b2
  /// @param cache_a2 输出: σ(z2)
  double ForwardWithCache(const Eigen::VectorXd & x,
                          Eigen::VectorXd & cache_h1, Eigen::VectorXd & cache_a1,
                          Eigen::VectorXd & cache_h2, Eigen::VectorXd & cache_a2) const;

  /// @brief 反向传播 (MSE loss)
  /// @param x 输入
  /// @param target 目标值
  /// @return 各层梯度
  Gradients Backward(const Eigen::VectorXd & x, double target) const;

  /// @brief 获取梯度 ∂y/∂x at point x (自动微分)
  Eigen::VectorXd GetGradient(const Eigen::VectorXd & x) const;

  /// @brief 获取 Hessian ∂²y/∂x² at point x (数值差分)
  /// @param x 查询点
  /// @param eps 差分步长
  Eigen::MatrixXd GetHessian(const Eigen::VectorXd & x, double eps = 1e-4) const;

  /// @brief 获取权重引用 (用于优化器更新)
  Weights & GetWeights() { return weights_; }
  const Weights & GetWeights() const { return weights_; }

  /// @brief 设置权重
  void SetWeights(const Weights & w) { weights_ = w; }

  /// @brief 获取维度
  int InputDim() const { return input_dim_; }
  int HiddenDim() const { return hidden_dim_; }
  int OutputDim() const { return output_dim_; }

private:
  int input_dim_, hidden_dim_, output_dim_;
  Weights weights_;
};

}  // namespace control_pkg

#endif