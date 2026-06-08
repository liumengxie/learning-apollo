#include "control_pkg/replay_buffer.hpp"
#include <algorithm>
#include <random>

namespace control_pkg
{

ReplayBuffer::ReplayBuffer(size_t capacity)
  : capacity_(capacity), buffer_(capacity)
{
  rng_ = std::mt19937(std::random_device{}());
}

void ReplayBuffer::Push(const Transition & t)
{
  buffer_[cursor_] = t;
  cursor_ = (cursor_ + 1) % capacity_;
  if (size_ < capacity_) size_++;
}

std::vector<ReplayBuffer::Transition> ReplayBuffer::Sample(size_t batch_size) const
{
  size_t n = std::min(batch_size, size_);
  std::vector<Transition> result;
  result.reserve(n);

  std::uniform_int_distribution<size_t> dist(0, size_ - 1);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(buffer_[dist(rng_)]);
  }
  return result;
}

}  // namespace control_pkg