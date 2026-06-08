#include "control_pkg/neural_network.hpp"
#include <cmath>
#include <algorithm>

namespace control_pkg
{

static double ReLU(double x) { return x > 0.0 ? x : 0.0; }
static double ReLUDerivative(double x) { return x > 0.0 ? 1.0 : 0.0; }

NeuralNetwork::NeuralNetwork(int input_dim, int hidden_dim, int output_dim)
  : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  // Xavier initialization
  double bound1 = std::sqrt(6.0 / (input_dim + hidden_dim));
  double bound2 = std::sqrt(6.0 / (hidden_dim + hidden_dim));
  double bound3 = std::sqrt(6.0 / (hidden_dim + output_dim));

  std::uniform_real_distribution<> d1(-bound1, bound1);
  std::uniform_real_distribution<> d2(-bound2, bound2);
  std::uniform_real_distribution<> d3(-bound3, bound3);

  weights_.W1 = Eigen::MatrixXd::NullaryExpr(hidden_dim, input_dim, [&]() { return d1(gen); });
  weights_.W2 = Eigen::MatrixXd::NullaryExpr(hidden_dim, hidden_dim, [&]() { return d2(gen); });
  weights_.W3 = Eigen::MatrixXd::NullaryExpr(output_dim, hidden_dim, [&]() { return d3(gen); });

  weights_.b1 = Eigen::VectorXd::Zero(hidden_dim);
  weights_.b2 = Eigen::VectorXd::Zero(hidden_dim);
  weights_.b3 = Eigen::VectorXd::Zero(output_dim);
}

double NeuralNetwork::Forward(const Eigen::VectorXd & x) const
{
  Eigen::VectorXd z1 = weights_.W1 * x + weights_.b1;
  Eigen::VectorXd a1 = z1.unaryExpr([](double v) { return ReLU(v); });
  Eigen::VectorXd z2 = weights_.W2 * a1 + weights_.b2;
  Eigen::VectorXd a2 = z2.unaryExpr([](double v) { return ReLU(v); });
  Eigen::VectorXd y = weights_.W3 * a2 + weights_.b3;
  return y(0);
}

double NeuralNetwork::ForwardWithCache(const Eigen::VectorXd & x,
                                       Eigen::VectorXd & cache_h1,
                                       Eigen::VectorXd & cache_a1,
                                       Eigen::VectorXd & cache_h2,
                                       Eigen::VectorXd & cache_a2) const
{
  cache_h1 = weights_.W1 * x + weights_.b1;
  cache_a1 = cache_h1.unaryExpr([](double v) { return ReLU(v); });
  cache_h2 = weights_.W2 * cache_a1 + weights_.b2;
  cache_a2 = cache_h2.unaryExpr([](double v) { return ReLU(v); });
  double y = (weights_.W3 * cache_a2 + weights_.b3)(0);
  return y;
}

NeuralNetwork::Gradients NeuralNetwork::Backward(const Eigen::VectorXd & x, double target) const
{
  Gradients grads;
  grads.dW1 = Eigen::MatrixXd::Zero(hidden_dim_, input_dim_);
  grads.dW2 = Eigen::MatrixXd::Zero(hidden_dim_, hidden_dim_);
  grads.dW3 = Eigen::MatrixXd::Zero(output_dim_, hidden_dim_);
  grads.db1 = Eigen::VectorXd::Zero(hidden_dim_);
  grads.db2 = Eigen::VectorXd::Zero(hidden_dim_);
  grads.db3 = Eigen::VectorXd::Zero(output_dim_);

  // Forward with cache
  Eigen::VectorXd h1, a1, h2, a2;
  double y_pred = ForwardWithCache(x, h1, a1, h2, a2);

  // MSE loss: L = 0.5 * (y_pred - target)^2
  double dL_dy = y_pred - target;   // ∂L/∂y

  // Layer 3 gradients (Linear)
  grads.dW3 = dL_dy * a2.transpose();               // (1 × hidden)
  grads.db3(0) = dL_dy;

  Eigen::VectorXd dL_da2 = weights_.W3.transpose() * Eigen::VectorXd::Constant(1, dL_dy);

  // Layer 2 gradients (ReLU)
  Eigen::VectorXd dL_dh2(hidden_dim_);
  for (int i = 0; i < hidden_dim_; ++i)
    dL_dh2(i) = dL_da2(i) * ReLUDerivative(h2(i));

  grads.dW2 = dL_dh2 * a1.transpose();
  grads.db2 = dL_dh2;

  Eigen::VectorXd dL_da1 = weights_.W2.transpose() * dL_dh2;

  // Layer 1 gradients (ReLU)
  Eigen::VectorXd dL_dh1(hidden_dim_);
  for (int i = 0; i < hidden_dim_; ++i)
    dL_dh1(i) = dL_da1(i) * ReLUDerivative(h1(i));

  grads.dW1 = dL_dh1 * x.transpose();
  grads.db1 = dL_dh1;

  return grads;
}

Eigen::VectorXd NeuralNetwork::GetGradient(const Eigen::VectorXd & x) const
{
  // 自动微分: 链式法则计算 ∂y/∂x
  Eigen::VectorXd h1 = weights_.W1 * x + weights_.b1;
  Eigen::VectorXd a1 = h1.unaryExpr([](double v) { return ReLU(v); });

  Eigen::VectorXd h2 = weights_.W2 * a1 + weights_.b2;
  Eigen::VectorXd a2 = h2.unaryExpr([](double v) { return ReLU(v); });

  // ∂y/∂a2 = W3'  (output_dim × hidden_dim → hidden_dim)
  Eigen::VectorXd dy_da2 = weights_.W3.transpose().row(0);

  // ∂a2/∂h2 = diag(ReLU'(h2))
  Eigen::VectorXd da2_dh2(hidden_dim_);
  for (int i = 0; i < hidden_dim_; ++i)
    da2_dh2(i) = ReLUDerivative(h2(i));

  Eigen::VectorXd dy_dh2 = dy_da2.cwiseProduct(da2_dh2);

  // ∂h2/∂a1 = W2'
  Eigen::MatrixXd dh2_da1 = weights_.W2.transpose();
  Eigen::VectorXd dy_da1 = dh2_da1 * dy_dh2;

  // ∂a1/∂h1 = diag(ReLU'(h1))
  Eigen::VectorXd da1_dh1(hidden_dim_);
  for (int i = 0; i < hidden_dim_; ++i)
    da1_dh1(i) = ReLUDerivative(h1(i));

  Eigen::VectorXd dy_dh1 = dy_da1.cwiseProduct(da1_dh1);

  // ∂h1/∂x = W1'
  Eigen::MatrixXd dh1_dx = weights_.W1.transpose();
  Eigen::VectorXd grad = dh1_dx * dy_dh1;

  return grad;
}

Eigen::MatrixXd NeuralNetwork::GetHessian(const Eigen::VectorXd & x, double eps) const
{
  // 数值差分: ∂²y/∂x_i∂x_j ≈ [f(x+e_i+e_j) - f(x+e_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)] / (4*eps²)
  int n = x.size();
  Eigen::MatrixXd H(n, n);

  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      Eigen::VectorXd x_pp = x, x_pm = x, x_mp = x, x_mm = x;
      double ei = (i == j) ? eps : eps / std::sqrt(2.0);
      double ej = (i == j) ? eps : eps / std::sqrt(2.0);

      x_pp(i) += ei; x_pp(j) += ej;
      x_pm(i) += ei; x_pm(j) -= ej;
      x_mp(i) -= ei; x_mp(j) += ej;
      x_mm(i) -= ei; x_mm(j) -= ej;

      double fpp = Forward(x_pp);
      double fpm = Forward(x_pm);
      double fmp = Forward(x_mp);
      double fmm = Forward(x_mm);

      H(i, j) = (fpp - fpm - fmp + fmm) / (4.0 * ei * ej);
      if (i != j) H(j, i) = H(i, j);
    }
  }
  return H;
}

}  // namespace control_pkg