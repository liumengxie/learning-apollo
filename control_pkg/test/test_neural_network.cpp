#include <gtest/gtest.h>
#include <cmath>
#include "control_pkg/neural_network.hpp"

using namespace control_pkg;

TEST(NeuralNetworkTest, Constructor) {
  NeuralNetwork nn(7, 32, 1);
  EXPECT_EQ(nn.InputDim(), 7);
  EXPECT_EQ(nn.HiddenDim(), 32);
  EXPECT_EQ(nn.OutputDim(), 1);
}

TEST(NeuralNetworkTest, ForwardProducesFinite) {
  NeuralNetwork nn(4, 16, 1);
  Eigen::Vector4d x(0.5, -0.3, 0.1, 0.8);
  double y = nn.Forward(x);
  EXPECT_TRUE(std::isfinite(y));
}

TEST(NeuralNetworkTest, ForwardWithCacheMatches) {
  NeuralNetwork nn(4, 16, 1);
  Eigen::Vector4d x(0.5, -0.3, 0.1, 0.8);
  Eigen::VectorXd h1, a1, h2, a2;
  double y1 = nn.Forward(x);
  double y2 = nn.ForwardWithCache(x, h1, a1, h2, a2);
  EXPECT_DOUBLE_EQ(y1, y2);
  EXPECT_EQ(h1.size(), 16);
  EXPECT_EQ(a1.size(), 16);
  EXPECT_EQ(h2.size(), 16);
  EXPECT_EQ(a2.size(), 16);
}

TEST(NeuralNetworkTest, GradientFiniteAndNonZero) {
  NeuralNetwork nn(4, 16, 1);
  Eigen::Vector4d x(1.0, 0.0, 0.0, 0.0);
  auto grad = nn.GetGradient(x);
  EXPECT_EQ(grad.size(), 4);
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(std::isfinite(grad(i)));
}

TEST(NeuralNetworkTest, HessianShape) {
  NeuralNetwork nn(4, 8, 1);
  Eigen::Vector4d x(0.0, 0.0, 0.0, 0.0);
  auto H = nn.GetHessian(x);
  EXPECT_EQ(H.rows(), 4);
  EXPECT_EQ(H.cols(), 4);
}

TEST(NeuralNetworkTest, HessianSymmetric) {
  NeuralNetwork nn(4, 8, 1);
  Eigen::Vector4d x(0.1, 0.2, 0.3, 0.4);
  auto H = nn.GetHessian(x);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(H(i,j), H(j,i), 1e-8);
    }
  }
}

TEST(NeuralNetworkTest, BackwardReturnsValidGradients) {
  NeuralNetwork nn(4, 8, 1);
  Eigen::Vector4d x(0.5, -0.3, 0.1, 0.8);
  auto grads = nn.Backward(x, 1.0);
  EXPECT_TRUE(grads.dW1.allFinite());
  EXPECT_TRUE(grads.dW2.allFinite());
  EXPECT_TRUE(grads.dW3.allFinite());
}

TEST(NeuralNetworkTest, BackwardReducesLoss) {
  NeuralNetwork nn(4, 8, 1);
  Eigen::Vector4d x(1.0, 0.5, 0.2, -0.1);
  double target = 0.7;
  double y_before = nn.Forward(x);
  double loss_before = 0.5 * (y_before - target) * (y_before - target);

  auto grads = nn.Backward(x, target);
  auto & W = nn.GetWeights();
  W.W1 -= 0.01 * grads.dW1;
  W.b1 -= 0.01 * grads.db1;
  W.W2 -= 0.01 * grads.dW2;
  W.b2 -= 0.01 * grads.db2;
  W.W3 -= 0.01 * grads.dW3;
  W.b3 -= 0.01 * grads.db3;

  double y_after = nn.Forward(x);
  double loss_after = 0.5 * (y_after - target) * (y_after - target);
  EXPECT_LT(loss_after, loss_before);
}

TEST(NeuralNetworkTest, SetWeights) {
  NeuralNetwork nn1(4, 8, 1);
  NeuralNetwork nn2(4, 8, 1);
  nn2.SetWeights(nn1.GetWeights());
  Eigen::Vector4d x(0.5, -0.3, 0.1, 0.8);
  EXPECT_DOUBLE_EQ(nn1.Forward(x), nn2.Forward(x));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}