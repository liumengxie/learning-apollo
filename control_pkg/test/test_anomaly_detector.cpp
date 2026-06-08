#include <gtest/gtest.h>
#include <cmath>
#include <control_pkg/anomaly_detector.hpp>

namespace control_pkg
{

class AnomalyDetectorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    detector_ = new AnomalyDetector(500.0, 2.0, 1.0);
  }

  void TearDown() override
  {
    delete detector_;
  }

  std::vector<TrajectoryPoint> CreateValidTrajectory()
  {
    std::vector<TrajectoryPoint> traj;
    for (int i = 0; i < 10; ++i) {
      TrajectoryPoint pt;
      pt.x = static_cast<double>(i);
      pt.y = 0.0;
      pt.z = 0.0;
      pt.yaw = 0.0;
      pt.velocity = 10.0;
      pt.curvature = 0.0;
      pt.relative_time = static_cast<double>(i) * 0.1;
      traj.push_back(pt);
    }
    return traj;
  }

  AnomalyDetector * detector_;
};

TEST_F(AnomalyDetectorTest, Constructor)
{
  // Creating with 300ms max_msg_age; check timeout logic works
  AnomalyDetector d(300.0, 2.0, 1.0);
  // 400ms age > 300ms max → should return false
  EXPECT_FALSE(d.CheckMessageTimeout(0.0, 0.4));
  // 200ms age < 300ms max → should return true
  EXPECT_TRUE(d.CheckMessageTimeout(0.0, 0.2));
}

TEST_F(AnomalyDetectorTest, CheckMessageTimeout_Normal)
{
  EXPECT_TRUE(detector_->CheckMessageTimeout(0.0, 0.4));
}

TEST_F(AnomalyDetectorTest, CheckMessageTimeout_Timeout)
{
  EXPECT_FALSE(detector_->CheckMessageTimeout(0.0, 0.6));
}

TEST_F(AnomalyDetectorTest, CheckMessageTimeout_Boundary)
{
  EXPECT_TRUE(detector_->CheckMessageTimeout(0.0, 0.5));
}

TEST_F(AnomalyDetectorTest, CheckLocalizationValid_Normal)
{
  EXPECT_TRUE(detector_->CheckLocalizationValid(1.0, 2.0, 0.5));
}

TEST_F(AnomalyDetectorTest, CheckLocalizationValid_NaN)
{
  EXPECT_FALSE(detector_->CheckLocalizationValid(NAN, 2.0, 0.5));
}

TEST_F(AnomalyDetectorTest, CheckLocalizationValid_Inf)
{
  EXPECT_FALSE(detector_->CheckLocalizationValid(1.0, INFINITY, 0.5));
}

TEST_F(AnomalyDetectorTest, CheckLocalizationValid_HeadingOutOfRange)
{
  EXPECT_FALSE(detector_->CheckLocalizationValid(1.0, 2.0, 10.0));
}

TEST_F(AnomalyDetectorTest, CheckTrajectoryValid_Normal)
{
  auto traj = CreateValidTrajectory();
  EXPECT_TRUE(detector_->CheckTrajectoryValid(traj));
}

TEST_F(AnomalyDetectorTest, CheckTrajectoryValid_TooFewPoints)
{
  std::vector<TrajectoryPoint> empty;
  EXPECT_FALSE(detector_->CheckTrajectoryValid(empty));

  std::vector<TrajectoryPoint> one_point;
  TrajectoryPoint pt;
  pt.x = 0.0; pt.y = 0.0; pt.z = 0.0;
  pt.yaw = 0.0; pt.velocity = 10.0; pt.curvature = 0.0; pt.relative_time = 0.0;
  one_point.push_back(pt);
  EXPECT_FALSE(detector_->CheckTrajectoryValid(one_point));

  std::vector<TrajectoryPoint> two_points = {pt, pt};
  EXPECT_TRUE(detector_->CheckTrajectoryValid(two_points));
}

TEST_F(AnomalyDetectorTest, CheckTrajectoryValid_InvalidPoint)
{
  auto traj = CreateValidTrajectory();
  traj[3].x = NAN;
  EXPECT_FALSE(detector_->CheckTrajectoryValid(traj));
}

TEST_F(AnomalyDetectorTest, CheckTrackingError_Normal)
{
  EXPECT_TRUE(detector_->CheckTrackingError(1.0, 0.5));
}

TEST_F(AnomalyDetectorTest, CheckTrackingError_Exceeded)
{
  EXPECT_FALSE(detector_->CheckTrackingError(3.0, 0.5));
  EXPECT_FALSE(detector_->CheckTrackingError(0.5, 1.5));
}

TEST_F(AnomalyDetectorTest, CheckTrackingError_Boundary)
{
  EXPECT_TRUE(detector_->CheckTrackingError(2.0, 1.0));
}

TEST_F(AnomalyDetectorTest, CheckTrackingError_NaN)
{
  EXPECT_FALSE(detector_->CheckTrackingError(NAN, 0.5));
}

TEST_F(AnomalyDetectorTest, Detect_AllNormal)
{
  auto traj = CreateValidTrajectory();
  auto status = detector_->Detect(0.0, 0.0, 0.0, 0.1, traj, 1.0, 2.0, 0.0, 0.5, 0.3);
  EXPECT_TRUE(status.is_ok);
}

TEST_F(AnomalyDetectorTest, Detect_TrajectoryTimeout)
{
  auto traj = CreateValidTrajectory();
  auto status = detector_->Detect(0.0, 0.0, 0.0, 1.0, traj, 1.0, 2.0, 0.0, 0.5, 0.3);
  EXPECT_FALSE(status.is_ok);
}

TEST_F(AnomalyDetectorTest, Detect_InvalidLocalization)
{
  auto traj = CreateValidTrajectory();
  auto status = detector_->Detect(0.0, 0.0, 0.0, 0.1, traj, NAN, 2.0, 0.0, 0.5, 0.3);
  EXPECT_FALSE(status.is_ok);
}

TEST_F(AnomalyDetectorTest, Detect_InvalidTrajectory)
{
  std::vector<TrajectoryPoint> empty;
  auto status = detector_->Detect(0.0, 0.0, 0.0, 0.1, empty, 1.0, 2.0, 0.0, 0.5, 0.3);
  EXPECT_FALSE(status.is_ok);
}

TEST_F(AnomalyDetectorTest, Detect_TrackingErrorExceeded)
{
  auto traj = CreateValidTrajectory();
  auto status = detector_->Detect(0.0, 0.0, 0.0, 0.1, traj, 1.0, 2.0, 0.0, 3.0, 1.5);
  EXPECT_FALSE(status.is_ok);
}

TEST_F(AnomalyDetectorTest, Detect_MultipleAnomalies)
{
  std::vector<TrajectoryPoint> empty;
  auto status = detector_->Detect(0.0, 0.0, 0.0, 1.0, empty, NAN, 2.0, 10.0, 3.0, 1.5);
  EXPECT_FALSE(status.is_ok);
  EXPECT_GE(static_cast<int>(status.warnings.size()), 2);
}

}  // namespace control_pkg

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}