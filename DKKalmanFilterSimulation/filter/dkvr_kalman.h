#pragma once

#include "Eigen/Dense"

namespace dkvr
{
    struct FilterConfiguration
    {
        Eigen::Vector3f noise_gyro, noise_accel, noise_mag;
        float uncert_ori;
        float uncert_accel, uncert_mag;
        Eigen::Vector3f accel_read, mag_read;
        float timestep;
        float lpf_cutoff_linear_accel;
        float lpf_cutoff_magnetic_dist;
    };

    struct NominalState
    {
        Eigen::Quaternionf orientation;
        Eigen::Vector3f linear_accel;
        Eigen::Vector3f magnetic_dist;
    };

    struct ErrorState
    {
        Eigen::Vector3f orientation;
        Eigen::Vector3f linear_accel;
        Eigen::Vector3f magnetic_dist;
    };

    using ErrorCovarianceMatrix = Eigen::Matrix<float, 9, 9>;
    using KalmanGainMatrix = Eigen::Matrix<float, 9, 6>;

    struct ControlVector
    {
        Eigen::Vector3f gyro;
    };

    struct Measurement
    {
        Eigen::Vector3f accel;
        Eigen::Vector3f mag;
    };

    class DKKalmanFilter
    {
    public:
        void Init(FilterConfiguration config);
        void Predict(ControlVector control);
        void Update(Measurement measurement);

        NominalState nominal_state() const { return nominal_; }
        ErrorState error_state() const { return error_; }
        ErrorCovarianceMatrix error_covar() const { return error_covar_; }
        KalmanGainMatrix kalman_gain() const { return kalman_gain_; }

    private:
        FilterConfiguration config_{};
        NominalState nominal_{};
        ErrorState error_{};
        
        ErrorCovarianceMatrix error_covar_{};
        KalmanGainMatrix kalman_gain_{};

        Eigen::Matrix<float, 9, 9> state_transition_matrix_{};
        Eigen::Matrix<float, 9, 9> state_transition_noise_{};

        Eigen::Matrix<float, 6, 9> observation_matrix_{};
        Eigen::Matrix<float, 6, 6> observation_noise_{};

        Measurement estimated_measurement_{};
        bool ignore_accel_ = false;
        bool ignore_mag_ = false;

        float lpfc_linear_accel = 0.5f;
        float lpfc_magnetic_dist = 0.5f;
        const Eigen::Vector3f gravity_ref_{ 0, 0, -1.0f };
        Eigen::Vector3f magnetic_ref_{ 1, 0, 0 };
    };

}   // namespace dkvr