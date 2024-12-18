#include "dkvr_kalman.h"

#include <cmath>

#include "Eigen/Dense"

//#define DKVR_DISABLE_THRESHOLD

namespace dkvr
{

    namespace
    {
        consteval float square(float x) { return x * x; }
        constexpr float kLinearAccelerationThreshold = square(0.2f);
        constexpr float kMagneticDisturbanceThreshold = square(0.2f);
        constexpr float kUncertTest = 0.5f;

        Eigen::Matrix3f SkewSymmetrize(Eigen::Vector3f vec)
        {
            return Eigen::Matrix3f
            {
                {       0, -vec.z(),  vec.y()},
                { vec.z(),        0, -vec.x()},
                {-vec.y(),  vec.x(),        0}
            };
        }
    }

    void DKKalmanFilter::Init(FilterConfiguration config)
    {
        // update config
        config_ = config;
        lpfc_linear_accel = expf(-2.0f * EIGEN_PI * config.lpf_cutoff_linear_accel * config.timestep);
        lpfc_magnetic_dist = expf(-2.0f * EIGEN_PI * config.lpf_cutoff_magnetic_dist * config.timestep);
        
        // normalize
        config.accel_read.normalize();
        config.mag_read.normalize();

        // rotate references to match gravity
        float cos = sqrtf(0.5f - 0.5f * config.accel_read.z());
        float sin = sqrtf(0.5f + 0.5f * config.accel_read.z());
        float size = sqrtf(powf(config.accel_read.x(), 2) + powf(config.accel_read.y(), 2));
        Eigen::Quaternionf quat{ cos, -config.accel_read.y() * sin / size, config.accel_read.x() * sin / size, 0 };
        Eigen::Vector3f mag_rotated = quat._transformVector(config.mag_read);

        // remove magnetic declination
        magnetic_ref_.x() = sqrtf(powf(mag_rotated.x(), 2) + powf(mag_rotated.y(), 2));
        magnetic_ref_.y() = 0;
        magnetic_ref_.z() = mag_rotated.z();

        // find initial orientation (TRIAD method)
        Eigen::Vector3f   gxm = gravity_ref_.cross(magnetic_ref_);
        Eigen::Vector3f gxgxm = gravity_ref_.cross(gxm);

        Eigen::Vector3f   axm = config.accel_read.cross(config.mag_read);
        Eigen::Vector3f axaxm = config.accel_read.cross(axm);
        
        Eigen::Matrix3f reference_matrix, body_matrix;
        reference_matrix << gravity_ref_, gxm.normalized(), gxgxm.normalized();
        body_matrix      << config.accel_read, axm.normalized(), axaxm.normalized();
        Eigen::Matrix3f rotation_matrix = body_matrix * reference_matrix.transpose();

        float w = 0.5f * sqrtf(rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2) + 1.0f);
        float fqw = 4.0f * w;
        Eigen::Quaternionf initial_orientation{
            w,
            (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / fqw,
            (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / fqw,
            (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / fqw
        };

        // reset state variables
        nominal_.orientation = initial_orientation;
        nominal_.linear_accel.setZero();
        nominal_.magnetic_dist.setZero();
        error_.orientation.setZero();
        error_.linear_accel.setZero();
        error_.magnetic_dist.setZero();
        error_covar_.setZero();

        // update calculation matrix
        state_transition_noise_.setZero();
        observation_noise_.setZero();
        for (Eigen::Index i = 0; i < 3; i++)
        {
            state_transition_noise_(i + 0, i + 0) = config.noise_gyro(i) * config.timestep * config.timestep;
            state_transition_noise_(i + 3, i + 3) = config.uncert_accel * config.timestep;
            state_transition_noise_(i + 6, i + 6) = config.uncert_mag * config.timestep;
            observation_noise_(i + 0, i + 0) = config.noise_accel(i);
            observation_noise_(i + 3, i + 3) = config.noise_mag(i);
        }
    }

    void DKKalmanFilter::Predict(ControlVector control)
    {
        // update nominal state
        control.gyro *= 0.5f * config_.timestep;
        Eigen::Quaternionf rotation { 1.0f, control.gyro.x(), control.gyro.y(), control.gyro.z()};

        nominal_.orientation *= rotation;
        nominal_.linear_accel *= lpfc_linear_accel;
        nominal_.magnetic_dist *= lpfc_magnetic_dist;

        // update state transition matrix
        Eigen::Matrix<float, 9, 9> transition_matrix = Eigen::Matrix<float, 9, 9>::Identity();
        transition_matrix.topLeftCorner<3, 3>() += SkewSymmetrize(control.gyro).transpose();

        // update priori state error covariance
        error_covar_ = (transition_matrix * error_covar_ * transition_matrix.transpose()).eval();
        error_covar_ += state_transition_noise_;

        // real stupid simple orientation uncert injection
        for (int i = 0; i < 3; i++)
            error_covar_(i, i) += config_.uncert_ori * config_.timestep;
    }

    void DKKalmanFilter::Update(Measurement measurement)
    {
        // predict linear accel and magnetic dist
        Eigen::Quaternionf inv_quat = nominal_.orientation.conjugate();
        Eigen::Vector3f expected_acc = measurement.accel - inv_quat._transformVector(gravity_ref_);
        Eigen::Vector3f expected_mag = measurement.mag - inv_quat._transformVector(magnetic_ref_);
        ignore_accel_ = expected_acc.squaredNorm() < kLinearAccelerationThreshold;
        ignore_mag_   = expected_mag.squaredNorm() < kMagneticDisturbanceThreshold;

#ifdef DKVR_DISABLE_THRESHOLD
        ignore_accel_ = true;
        ignore_mag_ = true;
#endif
        if (ignore_accel_) measurement.accel.normalize();
        if (ignore_mag_)   measurement.mag.normalize();


        // global frame estimation
        Eigen::Vector3f gf_linear_accel = gravity_ref_ - nominal_.linear_accel;
        Eigen::Vector3f gf_magnetic_dist = magnetic_ref_ + nominal_.magnetic_dist;
        
        // sensor frame estimation
        estimated_measurement_.accel = inv_quat._transformVector(gf_linear_accel);
        estimated_measurement_.mag = inv_quat._transformVector(gf_magnetic_dist);

        // update observation matrix
        observation_matrix_.setZero();
        observation_matrix_.topLeftCorner<3, 3>() = SkewSymmetrize(estimated_measurement_.accel * 0.5f);
        observation_matrix_.bottomLeftCorner<3, 3>() = SkewSymmetrize(estimated_measurement_.mag * 0.5f);

        Eigen::Matrix3f rotation_mat = inv_quat.toRotationMatrix();
        if (!ignore_accel_) observation_matrix_.block<3, 3>(0, 3) = -rotation_mat;
        if (!ignore_mag_)   observation_matrix_.block<3, 3>(3, 6) = rotation_mat;

        // compute kalman gain
        Eigen::Matrix<float, 6, 6> inv = observation_matrix_ * error_covar_ * observation_matrix_.transpose() + observation_noise_;
        kalman_gain_ = error_covar_ * observation_matrix_.transpose() * inv.inverse();

        // update error state
        Eigen::Vector<float, 6> error_measurement;
        error_measurement.block<3, 1>(0, 0) = measurement.accel - estimated_measurement_.accel;
        error_measurement.block<3, 1>(3, 0) = measurement.mag - estimated_measurement_.mag;
        Eigen::Vector<float, 9> observed_error = kalman_gain_ * error_measurement;
        error_.orientation = observed_error.block<3, 1>(0, 0);      // Eigens are not trivially copyable
        error_.linear_accel = observed_error.block<3, 1>(3, 0);
        error_.magnetic_dist = observed_error.block<3, 1>(6, 0);

        // update error covariance
        Eigen::Matrix<float, 9, 9> temp1 = (Eigen::Matrix<float, 9, 9>::Identity() - kalman_gain_ * observation_matrix_);
        Eigen::Matrix<float, 9, 9> temp2 = kalman_gain_ * observation_noise_ * kalman_gain_.transpose();
        error_covar_ = (temp1 * error_covar_ * temp1.transpose() + temp2).eval();

        // inject error to nominal
        Eigen::Quaternionf quat{ 1.0f, 0.5f * error_.orientation.x(), 0.5f * error_.orientation.y(), 0.5f * error_.orientation.z() };
        nominal_.orientation *= quat;
        nominal_.linear_accel += error_.linear_accel;
        nominal_.magnetic_dist += error_.magnetic_dist;
        nominal_.orientation.normalize();
    }


}   // namespace dkvr