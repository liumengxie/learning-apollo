#include <gtest/gtest.h>
#include <cmath>
#include <control_pkg/mpc_controller.hpp>

namespace control_pkg
{

class MPCControllerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    MPCController::Config config;
    config.prediction_horizon = 10;
    config.dt = 0.05;
    config.wheelbase = 2.8;
    controller_ = new MPCController(config);
  }

  void TearDown() override
  {
    delete controller_;
  }

  std::vector<TrajectoryPoint> CreateStraightLineTrajectory(int n = 50)
  {
    std::vector<TrajectoryPoint> traj;
    for (int i = 0; i < n; ++i) {
      TrajectoryPoint pt;
      pt.x = static_cast<double>(i) * 0.5;
      pt.y = 0.0;
      pt.z = 0.0;
      pt.yaw = 0.0;
      pt.velocity = 10.0;
      pt.curvature = 0.0;
      pt.relative_time = static_cast<double>(i) * 0.05;
      traj.push_back(pt);
    }
    return traj;
  }

  std::vector<TrajectoryPoint> CreateCurvedTrajectory(int n = 50)
  {
    std::vector<TrajectoryPoint> traj;
    double radius = 50.0;
    for (int i = 0; i < n; ++i) {
      double angle = static_cast<double>(i) * 0.02;
      TrajectoryPoint pt;
      pt.x = radius * std::sin(angle);
      pt.y = radius * (1.0 - std::cos(angle));
      pt.z = 0.0;
      pt.yaw = angle;
      pt.velocity = 10.0;
      pt.curvature = 1.0 / radius;
      pt.relative_time = static_cast<double>(i) * 0.05;
      traj.push_back(pt);
    }
    return traj;
  }

  MPCController * controller_;
};

TEST_F(MPCControllerTest, Constructor)
{
  // Default config, solve empty trajectory
  StateVec state = StateVec::Zero();
  std::vector<TrajectoryPoint> empty;
  auto result = controller_->Solve(state, empty);
  EXPECT_FALSE(result.success);
}

TEST_F(MPCControllerTest, GetReferenceTrajectory_StraightLine)
{
  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  auto ref = controller_->GetReferenceTrajectory(state, traj);
  EXPECT_EQ(static_cast<int>(ref.size()), 10);
  for (const auto & r : ref) {
    EXPECT_NEAR(r(1), 0.0, 1e-6);
    EXPECT_NEAR(r(2), 0.0, 1e-6);
  }
}

TEST_F(MPCControllerTest, GetReferenceTrajectory_ClosestPoint)
{
  auto traj = CreateStraightLineTrajectory();
  // state at (10, 0): the straight line has points at x = i*0.5
  // at i=20, x=10.0 → closest
  StateVec state;
  state << 10.0, 0.0, 0.0, 10.0;
  auto ref = controller_->GetReferenceTrajectory(state, traj);
  EXPECT_NEAR(ref[0](0), 10.0, 1.0);
  EXPECT_NEAR(ref[0](1), 0.0, 1e-6);
}

TEST_F(MPCControllerTest, GetReferenceTrajectory_EmptyTrajectory)
{
  StateVec state = StateVec::Zero();
  std::vector<TrajectoryPoint> empty;
  auto ref = controller_->GetReferenceTrajectory(state, empty);
  EXPECT_TRUE(ref.empty());
}

TEST_F(MPCControllerTest, Solve_StraightLine_OnPath)
{
  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  auto result = controller_->Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_NEAR(result.steering_angle, 0.0, 0.05);
  EXPECT_NEAR(result.acceleration, 0.0, 0.1);
}

TEST_F(MPCControllerTest, Solve_StraightLine_LateralOffset)
{
  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 1.0, 0.0, 10.0;
  auto result = controller_->Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_LT(result.steering_angle, 0.05);
}

TEST_F(MPCControllerTest, Solve_StraightLine_SpeedError)
{
  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 5.0;
  auto result = controller_->Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_GT(result.acceleration, -0.1);
}

TEST_F(MPCControllerTest, Solve_CurvedTrajectory)
{
  auto traj = CreateCurvedTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  auto result = controller_->Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_NE(result.steering_angle, 0.0);
}

TEST_F(MPCControllerTest, PredictTrajectory)
{
  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  auto result = controller_->Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_EQ(static_cast<int>(result.predicted_x.size()), 11);  // N+1
  EXPECT_DOUBLE_EQ(result.predicted_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result.predicted_y[0], 0.0);
}

TEST_F(MPCControllerTest, SteeringAngleConstraint)
{
  MPCController::Config config;
  config.prediction_horizon = 10;
  config.dt = 0.05;
  config.wheelbase = 2.8;
  config.max_steering_angle = 0.3;
  MPCController ctrl(config);

  auto traj = CreateCurvedTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  auto result = ctrl.Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_LE(result.steering_angle, 0.3);
}

TEST_F(MPCControllerTest, AccelerationConstraint)
{
  MPCController::Config config;
  config.prediction_horizon = 10;
  config.dt = 0.05;
  config.wheelbase = 2.8;
  config.max_acceleration = 2.0;
  config.max_deceleration = -4.0;
  MPCController ctrl(config);

  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 5.0;
  auto result = ctrl.Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_LE(result.acceleration, 2.0);
  EXPECT_GE(result.acceleration, -4.0);
}

TEST_F(MPCControllerTest, SolveQP_PositiveDefinite)
{
  Eigen::MatrixXd H(2, 2);
  H << 2.0, 0.0,
       0.0, 3.0;
  Eigen::VectorXd g(2);
  g << 1.0, -2.0;

  Eigen::VectorXd u = controller_->SolveQP(H, g);
  EXPECT_NEAR(u(0), -0.5, 1e-6);
  EXPECT_NEAR(u(1), 2.0 / 3.0, 1e-6);
}

TEST_F(MPCControllerTest, SolveQP_Identity)
{
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd g(3);
  g << 2.0, -4.0, 0.0;

  Eigen::VectorXd u = controller_->SolveQP(H, g);
  EXPECT_NEAR(u(0), -2.0, 1e-6);
  EXPECT_NEAR(u(1), 4.0, 1e-6);
  EXPECT_NEAR(u(2), 0.0, 1e-6);
}

TEST_F(MPCControllerTest, GetControlFromSolution)
{
  const int nu = 2;
  Eigen::VectorXd U(4);
  U << 0.3, 1.5, 0.2, 1.0;
  double steer, accel;
  controller_->GetControlFromSolution(U, steer, accel);
  EXPECT_DOUBLE_EQ(steer, 0.3);
  EXPECT_DOUBLE_EQ(accel, 1.5);
}

TEST_F(MPCControllerTest, ApplyConstraints)
{
  MPCController::Config config;
  config.prediction_horizon = 5;
  config.dt = 0.05;
  config.wheelbase = 2.8;
  config.max_steering_angle = 0.5;
  config.max_acceleration = 2.0;
  config.max_deceleration = -3.0;
  MPCController ctrl(config);

  Eigen::VectorXd U(10);
  // Out-of-range inputs
  U << 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0;
  ControlVec last_ctrl = ControlVec::Zero();
  ctrl.ApplyConstraints(U, last_ctrl);
  EXPECT_LE(U(0), 0.5);
  EXPECT_LE(U(1), 2.0);
  EXPECT_GE(U(1), -3.0);
}

TEST_F(MPCControllerTest, Reset)
{
  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  auto result1 = controller_->Solve(state, traj);
  ASSERT_TRUE(result1.success);

  controller_->Reset();
  auto result2 = controller_->Solve(state, traj);
  ASSERT_TRUE(result2.success);
  EXPECT_DOUBLE_EQ(result1.steering_angle, result2.steering_angle);
  EXPECT_DOUBLE_EQ(result1.acceleration, result2.acceleration);
}

TEST_F(MPCControllerTest, Solve_EmptyTrajectory)
{
  StateVec state = StateVec::Zero();
  std::vector<TrajectoryPoint> empty;
  auto result = controller_->Solve(state, empty);
  EXPECT_FALSE(result.success);
}

TEST_F(MPCControllerTest, SolveTimeReasonable)
{
  auto traj = CreateStraightLineTrajectory();
  StateVec state;
  state << 0.0, 0.0, 0.0, 10.0;
  auto result = controller_->Solve(state, traj);
  ASSERT_TRUE(result.success);
  EXPECT_GT(result.solve_time_ms, 0.0);
  EXPECT_LT(result.solve_time_ms, 100.0);
}

}  // namespace control_pkg

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}