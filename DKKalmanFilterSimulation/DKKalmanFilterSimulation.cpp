#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <filter/dkvr_kalman.h>
#include <math/math_const.h>
#include <sensor/imu_generator.h>

// global configuration
namespace
{
    constexpr float kNoiseStddevGyr[3] = { 0.000546, 0.000646, 0.000648 };
    constexpr float kNoiseStddevAcc[3] = { 0.00217,  0.00162,  0.00185 };
    constexpr float kNoiseStddevMag[3] = { 0.00656,  0.00294,  0.00308 };
    constexpr float kTimestep = 0.01f;
    constexpr float kMagneticDip = 15;
}

// generator configuration
namespace
{
    constexpr float kStddevAngularRate = 0.7854f;   // (rad/s / s)
    constexpr float kStddevLinearAccel = 0.01f;     // (m/s^2 / s)
    constexpr float kStddevMagneticDist = 0.01f;    // (nG    / s)
    constexpr float kGenLPFCutoffAngular = 20;      // (Hz)
    constexpr float kGenLPFCutoffAccel = 50;        // (Hz)
    constexpr float kGenLPFCutoffMag = 5;           // (Hz)
    constexpr float kGyroToleranceStdDev = 2.4;     // (%)
}

// filter configuration
namespace
{
    constexpr float kUncertOrientation = 0.01f * 0.01f * dkvr::kDegToRad * dkvr::kDegToRad;  // (rad/s)^2
    constexpr float kUncertLinearAccel  =  0.01f;                   // (m/s^2)^2
    constexpr float kUncertMagneticDist =  0.01f;                   // (nG)^2
    constexpr float kFilterLPFCutoffAccel = 50;                     // (Hz)
    constexpr float kFilterLPFCutoffMag = 5;                        // (Hz)
}

namespace
{

    Eigen::Vector3f QuaternionToYpr(const Eigen::Quaternionf& quat)
    {
        // roll (x-axis rotation)
        float sinr_cosp = 2 * (quat.w() * quat.x() + quat.y() * quat.z());
        float cosr_cosp = 1 - 2 * (quat.x() * quat.x() + quat.y() * quat.y());
        float roll = std::atan2f(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        float sinp = std::sqrt(1 + 2 * (quat.w() * quat.y() - quat.x() * quat.z()));
        float cosp = std::sqrt(1 - 2 * (quat.w() * quat.y() - quat.x() * quat.z()));
        float pitch = 2 * std::atan2f(sinp, cosp) - 3.1415926535f / 2;

        // yaw (z-axis rotation)
        float siny_cosp = 2 * (quat.w() * quat.z() + quat.x() * quat.y());
        float cosy_cosp = 1 - 2 * (quat.y() * quat.y() + quat.z() * quat.z());
        float yaw = std::atan2f(siny_cosp, cosy_cosp);

        yaw *= dkvr::kRadToDeg;
        pitch *= dkvr::kRadToDeg;
        roll *= dkvr::kRadToDeg;

        return Eigen::Vector3f{ yaw, pitch, roll };
    }

    // For pretty console write
    void PrintVector3(const Eigen::Vector3f& vec)
    {
        float val[3]{ vec.x(), vec.y(), vec.z() };
        int depth[3]{ 4, 4, 4 };

        for (int i = 0; i < 3; i++) 
        {
            if (val[i] < 0)
            {
                val[i] *= -1.0f;
                depth[i]--;
            }
            while (val[i] >= 10.0f)
            {
                depth[i]--;
                val[i] /= 10.0f;
            }
        }

        for (int i = 0; i < 3; i++)
        {
            std::stringstream ss;
            ss << std::fixed;
            for (int j = 0; j < depth[i]; j++)
                ss << ' ';
            ss << std::setprecision(6) << vec[i] << ' ';
            std::cout << ss.str() << " ";
        }
        std::cout << std::endl;
    }

    // stupid lazyness functions for writing to file
    void WriteVector3(std::ofstream& fout, const Eigen::Vector3f& vec)
    {
        fout << vec.x() << "," << vec.y() << "," << vec.z() << ",";
    }

    void WriteQuaternion(std::ofstream& fout, const Eigen::Quaternionf& quat)
    {
        fout << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << ",";
    }

}

constexpr int kCount = 1500;
constexpr bool kEnableEvent = true;
constexpr bool kRandomImpulse = false;

// export file
int main()
{
    // setup generator
    dkvr::IMUGenerator generator;
    dkvr::IMUGeneratorConfig gen_conf
    {
        .noise_stddev_gyro = Eigen::Vector3f(kNoiseStddevGyr),
        .noise_stddev_accel = Eigen::Vector3f(kNoiseStddevAcc),
        .noise_stddev_mag = Eigen::Vector3f(kNoiseStddevMag),
        .timestep = kTimestep,
        .gyro_tolerance_stddev = kGyroToleranceStdDev,
        .stddev_angular_rate = kStddevAngularRate,
        .stddev_linear_accel = kStddevLinearAccel,
        .stddev_magnetic_dist = kStddevMagneticDist,
        .lpf_cutoff_angular_rate = kGenLPFCutoffAngular,
        .lpf_cutoff_accel = kGenLPFCutoffAccel,
        .lpf_cutoff_mag = kGenLPFCutoffMag,
        .magnetic_dip = kMagneticDip
    };

    if (kEnableEvent)
    {
        using namespace dkvr;
        // getting close to refrigerator
        generator.AddEvent(GeneratorEvent{ 1,    1.0f,    0, Eigen::Vector3f{0,  0.5f, 0}, ValueType::kLinearAcceleration });
        generator.AddEvent(GeneratorEvent{ 1,    3.0f, 3.5f, Eigen::Vector3f{2, -1.9f, 1.2f}, ValueType::kMagneticDisturbance });
        generator.AddEvent(GeneratorEvent{ 1.5f, 1.0f,    0, Eigen::Vector3f{0, -0.5f, 0}, ValueType::kLinearAcceleration });

        // backflipping
        generator.AddEvent(GeneratorEvent{ 4,    0.6f, 0, Eigen::Vector3f{0, 360, 0}, ValueType::kRotation });
        generator.AddEvent(GeneratorEvent{ 4,    0.6f, 0, Eigen::Vector3f{0, 0, -1.0f}, ValueType::kLinearAcceleration });
        generator.AddEvent(GeneratorEvent{ 4.3f, 0.6f, 0, Eigen::Vector3f{0, 0,  1.0f}, ValueType::kLinearAcceleration });

        // getting away from refrigerator
        generator.AddEvent(GeneratorEvent{ 6,    1.0f, 0, Eigen::Vector3f{0.0f, -0.5f, 0.0f}, ValueType::kLinearAcceleration });
        generator.AddEvent(GeneratorEvent{ 6.5f, 1.0f, 0, Eigen::Vector3f{0.0f,  0.5f, 0.0f}, ValueType::kLinearAcceleration });

    }

    generator.Init(gen_conf);
    generator.Update();
    dkvr::IMUValues val = generator.imu();

    dkvr::FilterConfiguration filter_conf
    {
        .noise_gyro = {powf(kNoiseStddevGyr[0], 2), powf(kNoiseStddevGyr[1], 2), powf(kNoiseStddevGyr[2], 2)},
        .noise_accel = {powf(kNoiseStddevAcc[0], 2), powf(kNoiseStddevAcc[1], 2), powf(kNoiseStddevAcc[2], 2)},
        .noise_mag = {powf(kNoiseStddevMag[0], 2), powf(kNoiseStddevMag[1], 2), powf(kNoiseStddevMag[2], 2)},
        .uncert_ori = kUncertOrientation,
        .uncert_accel = kUncertLinearAccel,
        .uncert_mag = kUncertMagneticDist,
        .accel_read = val.accel,
        .mag_read = val.mag,
        .timestep = kTimestep,
        .lpf_cutoff_linear_accel = kFilterLPFCutoffAccel,
        .lpf_cutoff_magnetic_dist = kFilterLPFCutoffMag
    };
    dkvr::DKKalmanFilter filter;
    filter.Init(filter_conf);

    // initial state
    dkvr::StateVariable initial_state = generator.true_state();

    // open file
    std::ofstream fout("./result.csv");
    if (!fout.is_open())
    {
        std::cout << "File open failed." << std::endl;
        return 0;
    }

    // these are label of data. quick and dirty way
    fout << "True angular rate x,y,z,"
        << "True orientation w,x,y,z,"
        << "True linear acceleration x,y,z,"
        << "True magnetic disturbance x,y,z,"
        << "Estimated orientation w,x,y,z,"
        << "Estimated linear acceleration x,y,z,"
        << "Estimated magnetic disturbance x,y,z,"
        << "Orientation Error y, p, r,"
        << "Linear acceleration magnitude,"
        << "Magnetic disturbance magnitude,"
        << "Gyro readings x, y, z"
        << '\n';

    for (int i = 0; i < kCount; i++)
    {
        // generate IMU values
        if (kRandomImpulse)
            generator.ApplyRandomImpulse();
        generator.Update();
        dkvr::StateVariable true_state = generator.true_state();
        dkvr::IMUValues imu = generator.imu();

        // run filter
        filter.Predict(dkvr::ControlVector{ .gyro = imu.gyro });
        dkvr::NominalState priori_state = filter.nominal_state();

        filter.Update(dkvr::Measurement{ .accel = imu.accel, .mag = imu.mag });

        // get filter state
        dkvr::NominalState posteriori_state = filter.nominal_state();
        dkvr::ErrorState error = filter.error_state();
        dkvr::ErrorCovarianceMatrix error_covar = filter.error_covar();
        dkvr::KalmanGainMatrix kalman_gain = filter.kalman_gain();

        // true state
        WriteVector3(fout, true_state.angular_rate * dkvr::kRadToDeg);
        WriteQuaternion(fout, true_state.orientation);
        WriteVector3(fout, true_state.linear_accel);
        WriteVector3(fout, true_state.magnetic_dist);
        
        // nominal state
        WriteQuaternion(fout, posteriori_state.orientation);
        WriteVector3(fout, posteriori_state.linear_accel);
        WriteVector3(fout, posteriori_state.magnetic_dist);

        // orientation error in YPR
        Eigen::Vector3f gfree_ypr_error = QuaternionToYpr(posteriori_state.orientation * true_state.orientation.conjugate());
        WriteVector3(fout, gfree_ypr_error);

        // magnitude
        fout << true_state.linear_accel.norm() << ",";
        fout << true_state.magnetic_dist.norm() << ",";

        // gyro readings
        WriteVector3(fout, imu.gyro * dkvr::kRadToDeg);

        fout << "\n";

        // perc
        if (!(i % 100))
        {
            std::cout << i * 100.0f / kCount << "% done." << std::endl;
        }
    }

    fout.close();

    return 0;
}



//// console
//int main()
//{
//    // setup filter
//    dkvr::IMUGeneratorConfig gen_conf
//    {
//        .noise_stddev_gyro = Eigen::Vector3f(kNoiseStddevGyr),
//        .noise_stddev_accel = Eigen::Vector3f(kNoiseStddevAcc),
//        .noise_stddev_mag = Eigen::Vector3f(kNoiseStddevMag),
//        .timestep = kTimestep,
//        .stddev_angular_rate = kStddevAngularRate,
//        .stddev_linear_accel = kStddevLinearAccel,
//        .stddev_magnetic_dist = kStddevMagneticDist,
//        .lpf_cutoff_accel = kGenLPFCutoffAccel,
//        .lpf_cutoff_mag = kGenLPFCutoffMag,
//        .magnetic_dip = kMagneticDip
//    };
//
//    dkvr::IMUGenerator generator;
//    generator.Init(gen_conf);
//    generator.Update();
//    dkvr::IMUValues val = generator.imu();
//
//    dkvr::FilterConfiguration filter_conf
//    {
//        .noise_gyro = {powf(kNoiseStddevGyr[0], 2), powf(kNoiseStddevGyr[1], 2), powf(kNoiseStddevGyr[2], 2)},
//        .noise_accel = {powf(kNoiseStddevAcc[0], 2), powf(kNoiseStddevAcc[1], 2), powf(kNoiseStddevAcc[2], 2)},
//        .noise_mag = {powf(kNoiseStddevMag[0], 2), powf(kNoiseStddevMag[1], 2), powf(kNoiseStddevMag[2], 2)},
//        .uncert_accel = Eigen::Vector3f(kUncertLinearAccel),
//        .uncert_mag = Eigen::Vector3f(kUncertMagneticDist),
//        .accel_read = val.accel,
//        .mag_read = val.mag,
//        .timestep = kTimestep,
//        .lpf_cutoff_linear_accel = kFilterLPFCutoffAccel,
//        .lpf_cutoff_magnetic_dist = kFilterLPFCutoffMag
//    };
//    dkvr::DKKalmanFilter filter;
//    filter.Init(filter_conf);
//
//    // initial state
//    dkvr::StateVariable initial_state = generator.true_state();
//    std::cout << "Init Orientation : "; PrintVector3(QuaternionToYpr(initial_state.orientation));
//
//    int skip = 0;
//    while (true)
//    {
//        // generate IMU values
//        generator.Update();
//        dkvr::StateVariable true_state = generator.true_state();
//        dkvr::IMUValues imu = generator.imu();
//
//        // run filter
//        filter.Predict(dkvr::ControlVector{ .gyro = imu.gyro });
//        dkvr::NominalState priori_state = filter.nominal_state();
//
//        filter.Update(dkvr::Measurement{ .accel = imu.accel, .mag = imu.mag });
//
//        if (skip > 0)
//        {
//            skip--;
//            continue;
//        }
//
//        // get filter state
//        dkvr::NominalState posteriori_state = filter.nominal_state();
//        dkvr::ErrorState error = filter.error_state();
//        dkvr::ErrorCovarianceMatrix error_covar = filter.error_covar();
//        dkvr::KalmanGainMatrix kalman_gain = filter.kalman_gain();
//
//        std::cout << "True Orientation : "; PrintVector3(QuaternionToYpr(true_state.orientation));
//        std::cout << "          Priori : "; PrintVector3(QuaternionToYpr(priori_state.orientation));
//        std::cout << "      Posteriori : "; PrintVector3(QuaternionToYpr(posteriori_state.orientation));
//        std::cout << "           Error : "; PrintVector3(error.orientation);
//        std::cout << std::endl;
//
//        std::cout << "True Linear Acc. : "; PrintVector3(true_state.linear_accel);
//        std::cout << "          Priori : "; PrintVector3(priori_state.linear_accel);
//        std::cout << "      Posteriori : "; PrintVector3(posteriori_state.linear_accel);
//        std::cout << "           Error : "; PrintVector3(error.linear_accel);
//        std::cout << std::endl;
//
//        std::cout << "True Mag. Dist.  : "; PrintVector3(true_state.magnetic_dist);
//        std::cout << "          Priori : "; PrintVector3(priori_state.magnetic_dist);
//        std::cout << "      Posteriori : "; PrintVector3(posteriori_state.magnetic_dist);
//        std::cout << "           Error : "; PrintVector3(error.magnetic_dist);
//        std::cout << std::endl;
//
//        while (true)
//        {
//            std::string temp;
//            std::cin >> temp;
//            if (!temp.compare("exit"))
//                return 0;
//
//            if (!temp.compare("covar"))
//            {
//                std::cout << error_covar << std::endl;
//                continue;
//            }
//
//            if (!temp.compare("gain"))
//            {
//                std::cout << kalman_gain << std::endl;
//                continue;
//            }
//
//            if (!temp.compare("imu"))
//            {
//                std::cout << "Gyro  Measurement : "; PrintVector3(imu.gyro * kRadToDeg);
//                std::cout << "Accel Measurement : "; PrintVector3(imu.accel);
//                std::cout << "Mag   Measurement : "; PrintVector3(imu.mag);
//                std::cout << std::endl;
//                continue;
//            }
//
//            if (!temp.compare("zeroacc"))
//            {
//                generator.ResetAccel();
//                continue;
//            }
//
//            if (!temp.compare("zeromag"))
//            {
//                generator.ResetMag();
//                continue;
//            }
//
//            try
//            {
//                int num = std::atoi(temp.c_str());
//                if (num > 0)
//                {
//                    std::cout << "Skipping " << num << " times." << std::endl;
//                    skip = num;
//                    break;
//                }
//            }
//            catch (const std::exception& ex) {}
//
//            std::cout << "Unknown command." << std::endl;
//        }
//        
//    }
//    
//    return 0;
//}