#include <gtest/gtest.h>
#include "control_pkg/replay_buffer.hpp"

using namespace control_pkg;

TEST(ReplayBufferTest, Constructor) {
  ReplayBuffer buffer(100);
  EXPECT_EQ(buffer.Size(), 0u);
  EXPECT_FALSE(buffer.Full());
}

TEST(ReplayBufferTest, PushIncreasesSize) {
  ReplayBuffer buffer(100);
  ReplayBuffer::Transition t;
  t.state << 0, 0, 0, 0;
  t.action << 0, 0;
  t.reward = 1.0;
  t.next_state << 0.1, 0, 0, 10;
  t.done = false;

  buffer.Push(t);
  EXPECT_EQ(buffer.Size(), 1u);
  buffer.Push(t);
  EXPECT_EQ(buffer.Size(), 2u);
}

TEST(ReplayBufferTest, FullAtCapacity) {
  ReplayBuffer buffer(3);
  ReplayBuffer::Transition t;
  t.state << 0, 0, 0, 0; t.action << 0, 0;
  t.reward = 0; t.next_state << 0, 0, 0, 0; t.done = false;
  buffer.Push(t); buffer.Push(t); buffer.Push(t);
  EXPECT_TRUE(buffer.Full());
  buffer.Push(t);  // overwrites oldest
  EXPECT_EQ(buffer.Size(), 3u);
}

TEST(ReplayBufferTest, SampleReturnsCorrectCount) {
  ReplayBuffer buffer(100);
  ReplayBuffer::Transition t;
  t.state << 0, 0, 0, 0; t.action << 0, 0;
  t.reward = 0; t.next_state << 0, 0, 0, 0; t.done = false;
  for (int i = 0; i < 10; ++i) buffer.Push(t);
  auto batch = buffer.Sample(5);
  EXPECT_EQ(batch.size(), 5u);
}

TEST(ReplayBufferTest, SampleClampedToSize) {
  ReplayBuffer buffer(100);
  ReplayBuffer::Transition t;
  t.state << 0, 0, 0, 0; t.action << 0, 0;
  t.reward = 0; t.next_state << 0, 0, 0, 0; t.done = false;
  buffer.Push(t); buffer.Push(t);
  auto batch = buffer.Sample(10);
  EXPECT_EQ(batch.size(), 2u);
}

TEST(ReplayBufferTest, ClearResets) {
  ReplayBuffer buffer(100);
  ReplayBuffer::Transition t;
  t.state << 0, 0, 0, 0; t.action << 0, 0;
  t.reward = 0; t.next_state << 0, 0, 0, 0; t.done = false;
  for (int i = 0; i < 10; ++i) buffer.Push(t);
  buffer.Clear();
  EXPECT_EQ(buffer.Size(), 0u);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}