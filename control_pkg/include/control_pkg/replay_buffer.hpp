#ifndef CONTROL_PKG__REPLAY_BUFFER_HPP_
#define CONTROL_PKG__REPLAY_BUFFER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cstddef>

namespace control_pkg
{

/// @brief Experience Replay Buffer (FIFO, 固定容量)
///
/// 存储 (state, action, reward, next_state, done) 元组
/// 支持随机采样 batch 用于 TD-Learning
class ReplayBuffer
{
public:
  struct Transition
  {
    Eigen::Vector4d state;        // 4维
    Eigen::Vector2d action;       // 2维 [steer, accel]
    double reward;
    Eigen::Vector4d next_state;   // 4维
    bool done;
  };

  /// @param capacity 最大存储容量
  explicit ReplayBuffer(size_t capacity = 100000);

  /// @brief 存储一条 transition
  void Push(const Transition & t);

  /// @brief 随机采样 batch_size 条
  /// @return 采样结果; 若 buffer 不足, 全量返回
  std::vector<Transition> Sample(size_t batch_size) const;

  /// @brief 当前存储量
  size_t Size() const { return size_; }

  /// @brief 是否已满
  bool Full() const { return size_ >= capacity_; }

  /// @brief 清空
  void Clear() { buffer_.clear(); size_ = 0; cursor_ = 0; }

private:
  size_t capacity_;
  size_t size_{0};
  size_t cursor_{0};
  std::vector<Transition> buffer_;
  mutable std::mt19937 rng_;
};

}  // namespace control_pkg

#endif