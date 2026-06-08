#include <gtest/gtest.h>
#include <cmath>
#include <control_pkg/vehicle_model.hpp>

namespace control_pkg
{

class VehicleModelTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model_ = new VehicleModel(2.8);
  }

  void TearDown() override
  {
    delete model_;
  }

  VehicleModel * model_;
};

TEST_F(VehicleModelTest, Constructor)
{
  EXPECT_DOUBLE_EQ(model_->GetWheelbase(), 2.8);
}

TEST_F(VehicleModelTest, ZeroInputZeroState)
{
  StateVec state = StateVec::Zero();
  ControlVec control = ControlVec::Zero();
  StateVec next = model_->Update(state, control, 0.1);
  EXPECT_DOUBLE_EQ(next(0), 0.0);
  EXPECT_DOUBLE_EQ(next(1), 0.0);
  EXPECT_DOUBLE_EQ(next(2), 0.0);
  EXPECT_DOUBLE_EQ(next(3), 0.0);
}

TEST_F(VehicleModelTest, StraightLineMotion)
{
  StateVec state;
  state << 0.0, 0.0, 0.0, 5.0;
  ControlVec control = ControlVec::Zero();
  double dt = 1.0;
  StateVec next = model_->Update(state, control, dt);
  EXPECT_GT(next(0), 0.0);
  EXPECT_DOUBLE_EQ(next(1), 0.0);
  EXPECT_DOUBLE_EQ(next(2), 0.0);
  EXPECT_DOUBLE_EQ(next(3), 5.0);
}

TEST_F(VehicleModelTest, Acceleration)
{
  StateVec state;
  state << 0.0, 0.0, 0.0, 5.0;
  ControlVec control;
  control << 0.0, 2.0;
  double dt = 0.5;
  StateVec next = model_->Update(state, control, dt);
  EXPECT_DOUBLE_EQ(next(3), 6.0);
}

TEST_F(VehicleModelTest, SteeringMotion)
{
  StateVec state;
  state << 0.0, 0.0, 0.0, 3.0;
  ControlVec control;
  control << 0.3, 0.0;
  double dt = 1.0;
  StateVec next = model_->Update(state, control, dt);
  EXPECT_GT(next(2), 0.0);
}

TEST_F(VehicleModelTest, NegativeSteering)
{
  StateVec state;
  state << 0.0, 0.0, 0.0, 3.0;
  ControlVec control;
  control << -0.3, 0.0;
  double dt = 1.0;
  StateVec next = model_->Update(state, control, dt);
  EXPECT_LT(next(2), 0.0);
}

TEST_F(VehicleModelTest, Deceleration)
{
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  ControlVec control;
  control << 0.0, -3.0;
  double dt = 1.0;
  StateVec next = model_->Update(state, control, dt);
  EXPECT_DOUBLE_EQ(next(3), 7.0);
}

TEST_F(VehicleModelTest, NegativeVelocity)
{
  StateVec state;
  state << 0.0, 0.0, 0.0, -2.0;
  ControlVec control = ControlVec::Zero();
  double dt = 1.0;
  StateVec next = model_->Update(state, control, dt);
  EXPECT_LT(next(0), 0.0);
}

TEST_F(VehicleModelTest, LinearizeAtZero)
{
  StateVec state = StateVec::Zero();
  ControlVec control = ControlVec::Zero();
  double dt = 0.1;
  Eigen::Matrix4d A;
  Eigen::Matrix<double, 4, 2> B;

  model_->Linearize(state, control, dt, A, B);

  EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(A(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(A(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(A(0, 3), dt);

  EXPECT_NEAR(B(2, 0), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(B(3, 1), dt);
}

TEST_F(VehicleModelTest, LinearizeAtNonZero)
{
  StateVec state;
  state << 1.0, 2.0, 0.5, 5.0;
  ControlVec control;
  control << 0.1, 0.5;
  double dt = 0.1;
  Eigen::Matrix4d A;
  Eigen::Matrix<double, 4, 2> B;

  model_->Linearize(state, control, dt, A, B);

  EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(A(1, 1), 1.0);
  EXPECT_NE(B(2, 0), 0.0);
  EXPECT_DOUBLE_EQ(B(3, 1), dt);
}

TEST_F(VehicleModelTest, UpdateLinearizeConsistency)
{
  StateVec state;
  state << 2.0, 1.0, 0.2, 4.0;
  ControlVec control;
  control << 0.1, 0.3;
  double dt = 0.01;

  StateVec next_nonlinear = model_->Update(state, control, dt);

  Eigen::Matrix4d A;
  Eigen::Matrix<double, 4, 2> B;
  model_->Linearize(state, control, dt, A, B);

  StateVec next_linear = A * state + B * control;

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(next_nonlinear(i), next_linear(i), 0.05);
  }
}

}  // namespace control_pkg

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}