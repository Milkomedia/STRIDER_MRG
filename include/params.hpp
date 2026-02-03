#ifndef PARAMS_H
#define PARAMS_H

#include <chrono>
#include <Eigen/Dense>

/* ALL TUNABLE & CONFIGUABLE PARAMETERS ARE HERE */

namespace param {

// ===== Geometry control =====
// --- pos&att ctrl gain
static constexpr double kX[3] = {22.0, 22.0, 17.0}; // Position gain [x, y, z]
static constexpr double kV[3] = { 8.0,  8.0,  7.0}; // Velocity gain [x, y, z]
static constexpr double kR[3] = {10.0, 10.0,  3.0}; // Rotational gain [roll, pitch, yaw]
static constexpr double kW[3] = { 3.5,  3.5,  1.5}; // angular Velocity gain [roll, pitch, yaw]

static constexpr double kI  = 0.0;  // Attitude integral gain for roll and pitch
static constexpr double kyI = 0.0;  // Attitude integral gain for yaw
static constexpr double kIX = 23.0; // Position integral gains

// --- UAV Parameters ---
static constexpr double J[9] = {0.3, 0.0, 0.0,
                                0.0, 0.3, 0.0,
                                0.0, 0.0, 0.5318};
static constexpr double M  = 7.5;     // [kg]
static constexpr double G  = 9.80665; // [m/s^2] (must be positive)

// ===== Control Allocation =====
static constexpr double SERVO_DELAY_ALPHA = 0.093158;  // yaw trimming
static constexpr double SERVO_DELAY_BETA  = 1.0 - SERVO_DELAY_ALPHA; // this not tunable
static constexpr double TAUZ_MIN = -5.0; // saturation ref [Nm]
static constexpr double TAUZ_MAX =  5.0;

// ===== Thrust -> PWM model =====
static constexpr double PWM_A    = 46.5435;  // propeller thrust[N] = A * pwm^2 + B
static constexpr double PWM_B    = 8.6111;   // propeller thrust[N] = A * pwm^2 + B
static constexpr double PWM_ZETA = 0.02;     // propeller torque[Nm] = zeta * thrust
static constexpr double rotor_dir[4] = {1.0, -1.0, 1.0, -1.0}; // propeller torque direction

// ===== SBUS command =====
static constexpr double SBUS_X_RANGE       = 1.0;   // [m] k mapped to [-k, +k]
static constexpr double SBUS_Y_RANGE       = 1.0;   // [m] k mapped to [-k, +k]
static constexpr double SBUS_Z_RANGE       = 1.0;   // [m] k mapped to [0, +k]
static constexpr double SBUS_YAW_SPEED     = 20.0;  // [deg/s] @60Hz SBUS rate
static constexpr double SBUS_L_RANGE[2]    = { 0.45, 0.55};  // [m]
static constexpr double SBUS_COTZ_RANGE[2] = {-0.1, -0.25};  // [m]

// ===== OptiTrack offsets =====
static constexpr double OPTI_X_OFFSET  = 0.000; // [m]
static constexpr double OPTI_Y_OFFSET  = 0.645; // [m]

// ===== Butterworth cutoff frequencys =====
static constexpr double GYRO_Z_CUTOFF_HZ   = 5.0;
static constexpr double OPTI_VEL_CUTOFF_HZ = 4.0;

// ===== Control Frequencies =====
static constexpr std::chrono::steady_clock::duration CTRL_DT       = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(2700));
static constexpr std::chrono::steady_clock::duration MAX_PULL_TICK = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(800));
static constexpr std::chrono::steady_clock::duration MPC_DT        = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(5000));

// ===== Take-off parameters =====
static constexpr double IDLE_PWM_DUTY = 0.17; // override pwm duty [0.0~1.0]
static constexpr double TAKE_OFF_TIME = 4.0;  // smoothing thime   [sec]
static constexpr double INITIAL_RISING_COEFF = IDLE_PWM_DUTY / std::sqrt((M*G/4.0-PWM_B)/PWM_A) + 0.1; // this not tunable
static constexpr double RISING_COEFF_INC = (1.0 - INITIAL_RISING_COEFF) * std::chrono::duration_cast<std::chrono::duration<double>>(CTRL_DT).count() / TAKE_OFF_TIME; // this not tunable

// ===== Arm parameters =====
static constexpr double B2BASE_THETA[4] = {-0.25*M_PI, -0.75*M_PI, 0.75*M_PI, 0.25*M_PI};
static constexpr double B2BASE_ALPHA[4] = {M_PI, M_PI, M_PI, M_PI};
static constexpr double B2BASE_A[4]     = {0.120, 0.120, 0.120, 0.120};
static constexpr double DH_ARM_A[5]     = {0.1395, 0.115, 0.110, 0.024, 0.068};
static constexpr double DH_ARM_ALPHA[5] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, 0.0};

// ===== MPC parameters  =====
constexpr std::size_t N_STEPS  = 60;   // Steps per horizen
constexpr std::size_t NX       = 13;   // model state dim (include augmented state)
constexpr std::size_t NU_AUG   = 5;    // model augmented state dim
constexpr std::size_t NU       = 5;    // model input dim
constexpr std::size_t NP       = 11;   // model parameter dim

// ===== RT Scheduling =====
static constexpr int MAIN_PRIOR = 90;
static constexpr int T265_PRIOR = 82;
static constexpr int OPTI_PRIOR = 78;
static constexpr int DXL_PRIOR  = 74;
static constexpr int SBUS_PRIOR = 70;
static constexpr int MPC_PRIOR  = 60;

// ===== Sensor & Actuator hardcoded config =====
static constexpr const char* SBUS_PORT_NAME = "/dev/ttyUSB0";
static constexpr const char* DXL_PORT_NAME  = "/dev/ttyUSB1";
static constexpr const char* CAN_PORT_NAME  = "can0";
static constexpr const char* MOCAP_TYPE     = "optitrack";
static constexpr const char* OPTI_IP        = "192.168.10.115";
static constexpr const char* RIGIDBODY_NAME = "strider";
static constexpr const char* Log_File_NAME = "controller_log.bin";

} // namespace param

#endif // PARAMS_H