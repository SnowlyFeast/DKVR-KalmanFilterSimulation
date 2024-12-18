#pragma once

#include <list>
#include <queue>

#include "Eigen/Dense"

namespace dkvr
{

    struct IMUGeneratorConfig
    {
        Eigen::Vector3f noise_stddev_gyro;
        Eigen::Vector3f noise_stddev_accel;
        Eigen::Vector3f noise_stddev_mag;
        float timestep;
        float gyro_tolerance_stddev;
        float stddev_angular_rate;
        float stddev_linear_accel;
        float stddev_magnetic_dist;
        float lpf_cutoff_angular_rate, lpf_cutoff_accel, lpf_cutoff_mag;
        float magnetic_dip; // deg
    };

    struct StateVariable
    {
        Eigen::Quaternionf orientation;
        Eigen::Vector3f angular_rate;
        Eigen::Vector3f linear_accel;
        Eigen::Vector3f magnetic_dist;
    };

    struct IMUValues
    {
        Eigen::Vector3f gyro, accel, mag;
    };

    enum class ValueType
    {
        kRotation,
        kLinearAcceleration,
        kMagneticDisturbance
    };

    struct GeneratorEvent
    {
        float begin;
        float rising;
        float sustain;
        Eigen::Vector3f values;
        ValueType type;
    };

    class IMUGenerator
    {
    public:
        void Init(IMUGeneratorConfig config);
        void AddEvent(GeneratorEvent ev) { events_.push(ev); }
        void ApplyRandomImpulse();
        void Update();
        void ResetAccel() { state_.linear_accel.setZero(); }
        void ResetMag() { state_.magnetic_dist.setZero(); }
        
        StateVariable true_state() const { return state_; }
        IMUValues imu() const { return imu_; }

    private:
        void PopEvent();

        IMUGeneratorConfig config_;
        StateVariable state_;
        IMUValues imu_;
        std::queue<GeneratorEvent> events_;
        std::list<Eigen::Matrix3f> state_changes_;

        float internal_time_ = 0.0f;
        float lpfc_ang_rate_ = 0.5f;
        float lpfc_accel_ = 0.5f;
        float lpfc_mag_ = 0.5f;
        Eigen::Vector3f gravity_ref_{ 0.0f, 0.0f, -1.0f };
        Eigen::Vector3f magnetic_ref_{};

    };

}   // namespace dkvr