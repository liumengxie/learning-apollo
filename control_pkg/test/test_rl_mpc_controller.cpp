#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "control_pkg/rl_mpc_controller.hpp"

using namespace control_pkg;

class RLMPCControllerTest : public ::testing::Test {
protected:
  RLMPCController::Config config;

  void SetUp() override {
    config.mpc_config.prediction_horizon = 10;
    config.mpc_config.dt = 0.05;
    config.mpc_config.wheelbase = 2.8;
    config.vf_config.hidden_dim = 16;
    config.replay_buffer_capacity = 1000;
    config.train_interval = 5;
    config.train_batch_size = 16;
  }

  std::vector<TrajectoryPoint> straight_traj(int n = 50) {
    std::vector<TrajectoryPoint> t;
    for (int i = 0; i < n; ++i) {
      t.push_back({i*1.0, 0.0, 0.0, 0.0, 10.0, 0.0, i*0.1});
    }
    return t;
  }
};

TEST_F(RLMPCControllerTest, Constructor) {
  RLMPCController rl_mpc(config);
  EXPECT_EQ(rl_mpc.StepCount(), 0);
  EXPECT_EQ(rl_mpc.TrainCount(), 0);
}

TEST_F(RLMPCControllerTest, SolveReturnsSuccess) {
  RLMPCController rl_mpc(config);
  Eigen::Vector4d state(0, 0, 0, 10);
  auto traj = straight_traj();
  auto result = rl_mpc.Solve(state, traj, 0, 0, 0);
  EXPECT_TRUE(result.success);
}

TEST_F(RLMPCControllerTest, ComputeReward) {
  RLMPCController rl_mpc(config);
  double r = rl_mpc.ComputeReward(0.0, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(r, 0.0);

  double r2 = rl_mpc.ComputeReward(1.0, 0.0, 0.0);
  EXPECT_LT(r2, 0.0);
}

TEST_F(RLMPCControllerTest, StoreTransitionIncreasesBuffer) {
  RLMPCController rl_mpc(config);
  Eigen::Vector4d s(0,0,0,10);
  Eigen::Vector2d a(0.1, 0.5);
  rl_mpc.StoreTransition(s, a, -0.5, s, false);
  // Can't directly check buffer size without exposing it
  // Test via ShouldTrain
}

TEST_F(RLMPCControllerTest, ShouldTrainFalseAtStart) {
  RLMPCController rl_mpc(config);
  EXPECT_FALSE(rl_mpc.ShouldTrain());
}

TEST_F(RLMPCControllerTest, StepCountIncreases) {
  RLMPCController rl_mpc(config);
  Eigen::Vector4d state(0, 0, 0, 10);
  auto traj = straight_traj();
  rl_mpc.Solve(state, traj, 0, 0, 0);
  EXPECT_EQ(rl_mpc.StepCount(), 1);
  rl_mpc.Solve(state, traj, 0.1, 0.05, 0.2);
  EXPECT_EQ(rl_mpc.StepCount(), 2);
}

TEST_F(RLMPCControllerTest, ResetClearsState) {
  RLMPCController rl_mpc(config);
  Eigen::Vector4d state(0, 0, 0, 10);
  auto traj = straight_traj();
  rl_mpc.Solve(state, traj, 0, 0, 0);
  rl_mpc.Solve(state, traj, 0.1, 0, 0);
  rl_mpc.Reset();
  EXPECT_EQ(rl_mpc.StepCount(), 0);
}

TEST_F(RLMPCControllerTest, GetValueNetwork) {
  RLMPCController rl_mpc(config);
  auto & vn = rl_mpc.GetValueNetwork();
  Eigen::Matrix<double,7,1> s = Eigen::Matrix<double,7,1>::Zero();
  double v = vn.Predict(s);
  EXPECT_TRUE(std::isfinite(v));
}

TEST_F(RLMPCControllerTest, GetMPC) {
  RLMPCController rl_mpc(config);
  MPCController & mpc = rl_mpc.GetMPC();
  Eigen::Vector4d s(0,0,0,10);
  auto traj = straight_traj();
  auto result = mpc.Solve(s, traj);
  EXPECT_TRUE(result.success);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}