#include "imu_generator.h"

#include <cmath>
#include <list>
#include <vector>

#include "math/gaussian.h"
#include "math/math_const.h"

namespace dkvr
{

    namespace
    {
        Eigen::Vector3f RandomNormal3f()
        {
            return Eigen::Vector3f
            {
                Gaussian::NormalDistribution(),
                Gaussian::NormalDistribution(),
                Gaussian::NormalDistribution()
            };
        }
    }
   

    void IMUGenerator::Init(IMUGeneratorConfig config)
    {
        // deg to rad
        config_ = config;
        lpfc_ang_rate_ = expf(-2.0f * EIGEN_PI * config.lpf_cutoff_angular_rate * config.timestep);
        lpfc_accel_ = expf(-2.0f * EIGEN_PI * config.lpf_cutoff_accel * config.timestep);
        lpfc_mag_ = expf(-2.0f * EIGEN_PI * config.lpf_cutoff_mag * config.timestep);
        magnetic_ref_.x() = cosf(config.magnetic_dip * kDegToRad);
        magnetic_ref_.y() = 0.0f;
        magnetic_ref_.z() = sqrtf(1.0f - powf(magnetic_ref_.x(), 2));

        // init state
        state_.orientation = Eigen::Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
        state_.angular_rate.setZero();
        state_.linear_accel.setZero();
        state_.magnetic_dist.setZero();
    }

    void IMUGenerator::ApplyRandomImpulse()
    {
        // Wiener process
        Eigen::Vector3f new_ang = state_.angular_rate + RandomNormal3f() * config_.stddev_angular_rate * config_.timestep;
        Eigen::Vector3f new_accel = state_.linear_accel + RandomNormal3f() * config_.stddev_linear_accel * config_.timestep;
        Eigen::Vector3f new_mag = state_.magnetic_dist + RandomNormal3f() * config_.stddev_magnetic_dist * config_.timestep;

        // low-pass filter
        state_.angular_rate = lpfc_ang_rate_ * state_.angular_rate  + (1 - lpfc_ang_rate_) * new_ang;
        state_.linear_accel =  lpfc_accel_   * state_.linear_accel  + (1 - lpfc_accel_) * new_accel;
        state_.magnetic_dist = lpfc_mag_     * state_.magnetic_dist + (1 - lpfc_mag_) * new_mag;
    }

    void IMUGenerator::Update()
    {
        // update timestep and exec events
        internal_time_ += config_.timestep;
        PopEvent();

        if (!state_changes_.empty())
        {
            Eigen::Matrix3f& sc = state_changes_.front();
            state_.angular_rate += sc.col(0);
            state_.linear_accel += sc.col(1);
            state_.magnetic_dist += sc.col(2);
            state_changes_.pop_front();
        }

        // update state
        Eigen::Vector3f temp = state_.angular_rate * 0.5f * config_.timestep;
        Eigen::Quaternionf quat{ 1.0f, temp.x(), temp.y(), temp.z() };
        state_.orientation = (state_.orientation * quat).normalized();

        // generate imu
        Eigen::Quaternionf inv_quat = state_.orientation.conjugate();
        Eigen::Vector3f true_imu_accel = inv_quat._transformVector(gravity_ref_ - state_.linear_accel);
        Eigen::Vector3f true_imu_mag = inv_quat._transformVector(magnetic_ref_ + state_.magnetic_dist);
        float gyro_scale = (100.0f + Gaussian::NormalDistribution() * config_.gyro_tolerance_stddev) / 100.0f;
        imu_.gyro = state_.angular_rate * gyro_scale + RandomNormal3f().cwiseProduct(config_.noise_stddev_gyro);
        imu_.accel = true_imu_accel + RandomNormal3f().cwiseProduct(config_.noise_stddev_accel);
        imu_.mag = true_imu_mag + RandomNormal3f().cwiseProduct(config_.noise_stddev_mag);
    }

    void IMUGenerator::PopEvent()
    {
        if (events_.empty()) return;

        while (internal_time_ >= events_.front().begin)
        {
            // get event
            GeneratorEvent ev = events_.front();
            events_.pop();

            // generate gauss
            int sampling_rate = static_cast<int>(1.0f / config_.timestep);
            std::vector<float> gauss = Gaussian::GenerateGaussArray(sampling_rate, ev.rising, ev.type == ValueType::kRotation);

            // fill sustain
            int half = gauss.size() / 2;
            float peak = gauss[half];
            std::vector<float> sustained;
            for (int i = 0; i < half; i++)
                sustained.push_back(gauss[i]);
            for (int i = 0; i < ev.sustain * sampling_rate; i++)
                sustained.push_back(peak);
            for (int i = half; i < gauss.size(); i++)
                sustained.push_back(gauss[i]);

            // setup variable
            Eigen::Index idx = static_cast<Eigen::Index>(ev.type);
            auto iter = state_changes_.begin();
            bool end_of_iter = (iter == state_changes_.end());

            Eigen::Vector3f previous{ 0, 0, 0 };
            for (float f : sustained)
            {
                Eigen::Vector3f vector = f * ev.values;
                if (ev.type == ValueType::kRotation)
                    vector *= kDegToRad / config_.timestep;

                // derivative
                Eigen::Vector3f temp = vector;
                vector -= previous;
                previous = temp;

                // push back if iterator end
                if (end_of_iter)
                {
                    Eigen::Matrix3f mat = Eigen::Matrix3f::Zero();
                    mat.col(idx) = vector;
                    state_changes_.push_back(mat);
                }
                // or add to existing
                else
                {
                    iter->col(idx) += vector;
                    iter++;
                    if (iter == state_changes_.end())
                        end_of_iter = true;
                }
            }

            // last value
            if (end_of_iter)
            {
                Eigen::Matrix3f mat = Eigen::Matrix3f::Zero();
                mat.col(idx) -= previous;
                state_changes_.push_back(mat);
            }
            else
            {
                iter->col(idx) -= previous;
            }

            if (events_.empty()) break;
        }
    }


}   // namespace dkvr