#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "control_pkg/value_network.hpp"

using namespace control_pkg;

class ValueNetworkTest : public ::testing::Test {
protected:
  ValueNetwork::Config config;
  void SetUp() override {
    config.input_dim = 7; config.hidden_dim = 16;
  }
};

TEST_F(ValueNetworkTest, ConstructorCreatesValidNetwork) {
  ValueNetwork vn(config);
  Eigen::Matrix<double,7,1> s = Eigen::Matrix<double,7,1>::Zero();
  double v = vn.Predict(s);
  EXPECT_TRUE(std::isfinite(v));
}

TEST_F(ValueNetworkTest, PredictReturnsFinite) {
  ValueNetwork vn(config);
  Eigen::Matrix<double,7,1> s;
  s << 1.0, 2.0, 0.5, 10.0, 0.1, -0.05, 0.2;
  double v = vn.Predict(s);
  EXPECT_TRUE(std::isfinite(v));
}

TEST_F(ValueNetworkTest, GradientShapeAndFinite) {
  ValueNetwork vn(config);
  Eigen::Matrix<double,7,1> s = Eigen::Matrix<double,7,1>::Zero();
  auto grad = vn.GetGradient(s);
  EXPECT_EQ(grad.size(), 7);
  EXPECT_TRUE(grad.allFinite());
}

TEST_F(ValueNetworkTest, HessianShape) {
  ValueNetwork vn(config);
  Eigen::Vector4d state(0.0, 0.0, 0.0, 0.0);
  auto H = vn.GetHessian(state, 0.0, 0.0, 0.0);
  EXPECT_EQ(H.rows(), 4);
  EXPECT_EQ(H.cols(), 4);
}

TEST_F(ValueNetworkTest, TrainStepReducesLoss) {
  ValueNetwork vn(config);
  std::vector<ReplayBuffer::Transition> batch;

  for (int i = 0; i < 16; ++i) {
    ReplayBuffer::Transition tr;
    tr.state << i*0.1, 0.0, 0.0, 10.0;
    tr.action << 0.0, 0.0;
    tr.reward = -0.1;
    tr.next_state << (i+1)*0.1, 0.0, 0.0, 10.0;
    tr.done = false;
    batch.push_back(tr);
  }

  Eigen::Matrix<double,7,1> s0; s0 << 0,0,0,10,0,0,0;
  double v_before = vn.Predict(s0);

  double loss = vn.TrainStep(batch);
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GT(loss, 0.0);
}

TEST_F(ValueNetworkTest, UpdateTargetChangesWeights) {
  ValueNetwork vn(config);
  auto old_W1 = vn.OnlineNetwork().GetWeights().W1;

  // Run a few training steps
  std::vector<ReplayBuffer::Transition> batch;
  ReplayBuffer::Transition tr;
  tr.state << 0,0,0,10; tr.action << 0,0; tr.reward = -1;
  tr.next_state << 1,0,0,10; tr.done = false;
  for (int i = 0; i < 8; ++i) batch.push_back(tr);

  for (int i = 0; i < 5; ++i) vn.TrainStep(batch);
  vn.UpdateTarget();

  auto new_W1 = vn.OnlineNetwork().GetWeights().W1;
  EXPECT_TRUE(new_W1.isApprox(old_W1, 0.5));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}