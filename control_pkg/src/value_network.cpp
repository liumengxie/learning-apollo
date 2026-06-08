#include "control_pkg/value_network.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

namespace control_pkg
{

ValueNetwork::ValueNetwork(const Config & config)
  : config_(config),
    online_(config.input_dim, config.hidden_dim, 1),
    target_(config.input_dim, config.hidden_dim, 1)
{
  // target 初始化为 online 的副本
  target_.SetWeights(online_.GetWeights());

  // 初始化 Adam 动量
  auto zero_grad = [&]() {
    NeuralNetwork::Gradients g;
    int h = config.hidden_dim, in = config.input_dim;
    g.dW1 = Eigen::MatrixXd::Zero(h, in);
    g.dW2 = Eigen::MatrixXd::Zero(h, h);
    g.dW3 = Eigen::MatrixXd::Zero(1, h);
    g.db1 = Eigen::VectorXd::Zero(h);
    g.db2 = Eigen::VectorXd::Zero(h);
    g.db3 = Eigen::VectorXd::Zero(1);
    return g;
  };
  m_ = zero_grad();
  v_ = zero_grad();
}

Eigen::Matrix<double, 7, 1> ValueNetwork::BuildState(
    const Eigen::Vector4d & state, double lat_err,
    double head_err, double vel_err) const
{
  Eigen::Matrix<double, 7, 1> s;
  s << state(0), state(1), state(2), state(3), lat_err, head_err, vel_err;
  return s;
}

double ValueNetwork::Predict(const Eigen::VectorXd & state) const
{
  return online_.Forward(state);
}

Eigen::VectorXd ValueNetwork::GetGradient(const Eigen::VectorXd & state) const
{
  return online_.GetGradient(state);
}

Eigen::Matrix4d ValueNetwork::GetHessian(const Eigen::Vector4d & state,
                                         double lat_err, double head_err,
                                         double vel_err) const
{
  // 在7维增强状态上计算Hessian, 提取前4×4子矩阵
  auto s = BuildState(state, lat_err, head_err, vel_err);
  Eigen::MatrixXd H7 = online_.GetHessian(s);
  return H7.topLeftCorner<4, 4>();
}

double ValueNetwork::TrainStep(const std::vector<ReplayBuffer::Transition> & batch)
{
  if (batch.empty()) return 0.0;

  train_step_count_++;
  double total_loss = 0.0;

  // 累积梯度
  NeuralNetwork::Gradients sum_grads;
  int h = config_.hidden_dim, in = config_.input_dim;
  sum_grads.dW1 = Eigen::MatrixXd::Zero(h, in);
  sum_grads.dW2 = Eigen::MatrixXd::Zero(h, h);
  sum_grads.dW3 = Eigen::MatrixXd::Zero(1, h);
  sum_grads.db1 = Eigen::VectorXd::Zero(h);
  sum_grads.db2 = Eigen::VectorXd::Zero(h);
  sum_grads.db3 = Eigen::VectorXd::Zero(1);

  for (const auto & tr : batch)
  {
    // 构建7维增强状态
    // 简化: 不使用完整增强状态，用4维+误差
    // 实际: 这里需要lat/head/vel error, transition中无法直接获得
    // 用4维状态+零填充误差 → 仅用于演示，完整版需扩展Transition
    Eigen::Matrix<double, 7, 1> s_now;
    s_now << tr.state(0), tr.state(1), tr.state(2), tr.state(3), 0, 0, 0;

    Eigen::Matrix<double, 7, 1> s_next;
    s_next << tr.next_state(0), tr.next_state(1), tr.next_state(2), tr.next_state(3), 0, 0, 0;

    // TD target
    double v_next = tr.done ? 0.0 : target_.Forward(s_next);
    double target_val = tr.reward + config_.gamma * v_next;

    // 单个样本的梯度
    auto grads = online_.Backward(s_now, target_val);
    double pred = online_.Forward(s_now);
    total_loss += 0.5 * (pred - target_val) * (pred - target_val);

    // 累加
    sum_grads.dW1 += grads.dW1;
    sum_grads.dW2 += grads.dW2;
    sum_grads.dW3 += grads.dW3;
    sum_grads.db1 += grads.db1;
    sum_grads.db2 += grads.db2;
    sum_grads.db3 += grads.db3;
  }

  double n = static_cast<double>(batch.size());

  // 平均梯度
  sum_grads.dW1 /= n;
  sum_grads.dW2 /= n;
  sum_grads.dW3 /= n;
  sum_grads.db1 /= n;
  sum_grads.db2 /= n;
  sum_grads.db3 /= n;

  // 梯度裁剪
  auto clip = [&](Eigen::MatrixXd & m) {
    for (int i = 0; i < m.rows(); ++i)
      for (int j = 0; j < m.cols(); ++j)
        m(i,j) = std::clamp(m(i,j), -config_.grad_clip, config_.grad_clip);
  };
  auto clip_v = [&](Eigen::VectorXd & v) {
    for (int i = 0; i < v.size(); ++i)
      v(i) = std::clamp(v(i), -config_.grad_clip, config_.grad_clip);
  };
  clip(sum_grads.dW1); clip(sum_grads.dW2); clip(sum_grads.dW3);
  clip_v(sum_grads.db1); clip_v(sum_grads.db2); clip_v(sum_grads.db3);

  // Adam update
  double lr = config_.learning_rate;
  double b1 = beta1_, b2 = beta2_;
  double t = static_cast<double>(train_step_count_);
  double b1_corr = 1.0 - std::pow(b1, t);
  double b2_corr = 1.0 - std::pow(b2, t);
  double lr_t = lr * std::sqrt(b2_corr) / b1_corr;

  auto & W = online_.GetWeights();

  // W1
  m_.dW1 = b1 * m_.dW1.array() + (1 - b1) * sum_grads.dW1.array();
  v_.dW1 = b2 * v_.dW1.array() + (1 - b2) * sum_grads.dW1.array().square();
  W.W1 -= lr_t * (m_.dW1.array() / (v_.dW1.array().sqrt() + epsilon_)).matrix()
          + config_.weight_decay * W.W1;

  m_.db1 = b1 * m_.db1.array() + (1 - b1) * sum_grads.db1.array();
  v_.db1 = b2 * v_.db1.array() + (1 - b2) * sum_grads.db1.array().square();
  W.b1 -= lr_t * (m_.db1.array() / (v_.db1.array().sqrt() + epsilon_)).matrix();

  // W2
  m_.dW2 = b1 * m_.dW2.array() + (1 - b1) * sum_grads.dW2.array();
  v_.dW2 = b2 * v_.dW2.array() + (1 - b2) * sum_grads.dW2.array().square();
  W.W2 -= lr_t * (m_.dW2.array() / (v_.dW2.array().sqrt() + epsilon_)).matrix()
          + config_.weight_decay * W.W2;

  m_.db2 = b1 * m_.db2.array() + (1 - b1) * sum_grads.db2.array();
  v_.db2 = b2 * v_.db2.array() + (1 - b2) * sum_grads.db2.array().square();
  W.b2 -= lr_t * (m_.db2.array() / (v_.db2.array().sqrt() + epsilon_)).matrix();

  // W3
  m_.dW3 = b1 * m_.dW3.array() + (1 - b1) * sum_grads.dW3.array();
  v_.dW3 = b2 * v_.dW3.array() + (1 - b2) * sum_grads.dW3.array().square();
  W.W3 -= lr_t * (m_.dW3.array() / (v_.dW3.array().sqrt() + epsilon_)).matrix()
          + config_.weight_decay * W.W3;

  m_.db3 = b1 * m_.db3.array() + (1 - b1) * sum_grads.db3.array();
  v_.db3 = b2 * v_.db3.array() + (1 - b2) * sum_grads.db3.array().square();
  W.b3 -= lr_t * (m_.db3.array() / (v_.db3.array().sqrt() + epsilon_)).matrix();

  return total_loss / n;
}

void ValueNetwork::UpdateTarget()
{
  auto & O = online_.GetWeights();
  auto & T = target_.GetWeights();
  double tau = config_.tau;

  T.W1 = tau * O.W1 + (1.0 - tau) * T.W1;
  T.W2 = tau * O.W2 + (1.0 - tau) * T.W2;
  T.W3 = tau * O.W3 + (1.0 - tau) * T.W3;
  T.b1 = tau * O.b1 + (1.0 - tau) * T.b1;
  T.b2 = tau * O.b2 + (1.0 - tau) * T.b2;
  T.b3 = tau * O.b3 + (1.0 - tau) * T.b3;
}

void ValueNetwork::SaveWeights(const std::string & filepath) const
{
  // 简化实现: 保存到文本文件
  std::ofstream ofs(filepath);
  if (!ofs) return;
  auto & W = online_.GetWeights();
  ofs << W.W1.rows() << " " << W.W1.cols() << "\n" << W.W1 << "\n";
  ofs << W.W2.rows() << " " << W.W2.cols() << "\n" << W.W2 << "\n";
  ofs << W.W3.rows() << " " << W.W3.cols() << "\n" << W.W3 << "\n";
  ofs << W.b1.size() << "\n" << W.b1 << "\n";
  ofs << W.b2.size() << "\n" << W.b2 << "\n";
  ofs << W.b3.size() << "\n" << W.b3 << "\n";
}

void ValueNetwork::LoadWeights(const std::string & filepath)
{
  std::ifstream ifs(filepath);
  if (!ifs) return;
  auto & W = online_.GetWeights();
  int r, c;

  ifs >> r >> c; W.W1.resize(r, c);
  for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) ifs >> W.W1(i, j);

  ifs >> r >> c; W.W2.resize(r, c);
  for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) ifs >> W.W2(i, j);

  ifs >> r >> c; W.W3.resize(r, c);
  for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) ifs >> W.W3(i, j);

  ifs >> r; W.b1.resize(r);
  for (int i = 0; i < r; ++i) ifs >> W.b1(i);

  ifs >> r; W.b2.resize(r);
  for (int i = 0; i < r; ++i) ifs >> W.b2(i);

  ifs >> r; W.b3.resize(r);
  for (int i = 0; i < r; ++i) ifs >> W.b3(i);

  target_.SetWeights(W);
}

}  // namespace control_pkg