#ifndef PARAMS_H
#define PARAMS_H

#include <chrono>
#include <Eigen/Dense>

/* ALL TUNABLE & CONFIGUABLE PARAMETERS ARE HERE */

namespace param {

// ===== Path planning Parameters =====
static inline const Eigen::Vector3d DEFAULT_pos     = Eigen::Vector3d(0.0, -1.0, 1.00);
static inline const Eigen::Vector3d DEFAULT_r_cot   = Eigen::Vector3d(0.0,  0.0, 0.24);
static inline const Eigen::Vector3d DEFAULT_heading = Eigen::Vector3d(1.0,  0.0, 0.00);
static constexpr double DEFAULT_l                   = 0.48;

// ----------------------------------*-*-*-*-*-*-*-*-*-----------P_R
//                               *   |
//                         *         |  p(t) = p0 + Δp * (1 - cos(π t / T)) / 2
//                     *             |  v(t) = Δp * (π / (2T)) * sin(π t / T)
//                 *                 |  a(t) = Δp * (π² / (2T²)) * cos(π t / T)
//              *                    |
//            *                      |
// -*-*-*-*-*<----- PATH_T_MOVE ---->-----PATH_SETTLE_MAX---->---P_L

static inline const Eigen::Vector3d P_L             = Eigen::Vector3d(0.0, -1.0, 1.0);
static inline const Eigen::Vector3d P_R             = Eigen::Vector3d(0.0,  1.0, 1.0);

static constexpr double DEFAULT_POS_TOL             = 0.08; // [m]
static constexpr double DEFAULT_RCOT_TOL            = 0.01; // [m]

static constexpr double PATH_T_MOVE                 = 3.00;
static constexpr double PATH_SETTLE_MAX             = 5.00;

enum class PathStage : uint8_t {
  HOLD_LEFT   = 0,
  MOVE_L2R    = 1,
  HOLD_RIGHT  = 2,
  MOVE_R2L    = 3,
};

// ===== Geometry position control gain =====
static constexpr double kX[3]  = {28.5, 28.5, 26.00};    // Position gain [x, y, z]
static constexpr double kV[3]  = {18.0, 18.0, 17.00};    // Velocity gain [x, y, z]
static constexpr double kIX[3] = {15.0, 15.0, 20.0};     // Integral gain [x, y, z]

// ===== Geometry attitude control gain =====
static constexpr double kR[3]  = {50.0, 46.0,  6.5}; // Rotational gain [roll, pitch, yaw]
static constexpr double kW[3]  = {11.0, 11.0,  3.0}; // angular Velocity gain [roll, pitch, yaw]
static constexpr double kI     = 0.00;  // Integral gain for roll and pitch -> gaseggi しんで
static constexpr double kyI    = 1.20;  // Integral gain for yaw

// ===== UAV Parameter =====
static constexpr double J[9] = {    0.3, -0.0006, -0.0006,
                                -0.0006,     0.3,  0.0006,
                                -0.0006,  0.0006,  0.5318}; // [kg m^2]
static constexpr double M  = 7.25;    // [kg]
static constexpr double G  = 9.80665; // [m/s^2] (must be positive)

// ===== Control Allocation =====
static constexpr double SERVO_DELAY_ALPHA = 0.093158;  // yaw trimming
static constexpr double SERVO_DELAY_BETA  = 1.0 - SERVO_DELAY_ALPHA; // this not tunable

// ===== Butterworth cutoff frequencys =====
static constexpr double GYRO_XY_CUTOFF_HZ  = 10.0;
static constexpr double GYRO_Z_CUTOFF_HZ   = 5.0;
static constexpr double OPTI_VEL_CUTOFF_HZ = 4.0;

// ===== Various saturation parameters =====
static constexpr double EX_NORM_MAX         = 2.0;  // position control, position error max [m]
static constexpr double kIX_SAT[3]          = {15.0, 15.0, 20.0}; // position control, integral max on x,y,z [N]

static constexpr double ROLL_TORQUE_SAT     = 5.0;  // attitude control, torque max [Nm]
static constexpr double PITCH_TORQUE_SAT    = 5.0;  // attitude control, torque max [Nm]
static constexpr double YAW_TORQUE_SAT      = 5.0;  // attitude control, torque max [Nm]
static constexpr double ER_NORM_MAX         = 50.0 * M_PI / 180.0; // attitude control, attitude error max [rad]

static constexpr double MINIMUM_THRUST_SAT  = 8.7;  // sequential control allocation, thrust of each prop (min 5%) [N]
static constexpr double MAXIMUM_THRUST_SAT  = 53.3; // sequential control allocation, thrust of each prop (max 98%) [N]
static constexpr double REACTION_TORQUE_SAT = 5.0;  // sequential control allocation, reaction torque max [Nm]
static constexpr double TILT_ANGLE_SAT      = 25.0 * M_PI / 180.0; // sequential control allocation, tilt angle max [rad]

// ===== SBUS command =====
static constexpr double SBUS_X_RANGE       = 1.0;   // [m] k mapped to [-k, +k]
static constexpr double SBUS_Y_RANGE       = 1.0;   // [m] k mapped to [-k, +k]
static constexpr double SBUS_Z_RANGE       = 1.0;   // [m] k mapped to [ 0, -k]
static constexpr double SBUS_YAW_SPEED     = 20.0;  // [deg/s] @60Hz SBUS rate
static constexpr double SBUS_L_RANGE[2]    = { 0.46, 0.50};     // [m]
static constexpr double SBUS_COTZ_RANGE[2] = {-0.21, -0.27};    // [m]
static constexpr double SBUS_COTXY_RANGE[2] = {-0.047, 0.047};  // [m]

// ===== MPC parameters  =====
constexpr std::size_t N_STEPS  = 60;   // Steps per horizen
constexpr std::size_t NX       = 13;   // model state dim (include augmented state)
constexpr std::size_t NU_AUG   = 5;    // model augmented state dim
constexpr std::size_t NU       = 5;    // model input dim
constexpr std::size_t NP       = 11;   // model parameter dim

// ===== OptiTrack offsets =====
static constexpr double OPTI_X_OFFSET  = 0.000; // [m]
static constexpr double OPTI_Y_OFFSET  = 0.645; // [m]

// ===== Control Frequencies =====
static constexpr std::chrono::steady_clock::duration CTRL_DT       = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(2700)); // ~370Hz
static constexpr std::chrono::steady_clock::duration MAX_PULL_TICK = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(800));
static constexpr std::chrono::steady_clock::duration MPC_DT        = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(5000)); // 200Hz

// ===== Thrust -> PWM model =====
static constexpr double PWM_A    = 46.5435;  // propeller thrust[N] = A * pwm^2 + B
static constexpr double PWM_B    = 8.6111;   // propeller thrust[N] = A * pwm^2 + B
static constexpr double PWM_ZETA = 0.02;     // propeller torque[Nm] = zeta * thrust
static constexpr double rotor_dir[4] = {1.0, -1.0, 1.0, -1.0}; // propeller torque direction

// ===== Take-off parameters =====
static constexpr double IDLE_PWM_DUTY = 0.15; // override pwm duty [0.0~1.0]
static constexpr double TAKE_OFF_TIME = 3.0;  // smoothing thime   [sec]
static constexpr double INITIAL_RISING_COEFF = IDLE_PWM_DUTY / std::sqrt((M*G/4.0-PWM_B)/PWM_A) + 0.1; // this not tunable
static constexpr double RISING_COEFF_INC = (1.0 - INITIAL_RISING_COEFF) * std::chrono::duration_cast<std::chrono::duration<double>>(CTRL_DT).count() / TAKE_OFF_TIME; // this not tunable

// ===== Arm parameters =====
static constexpr double B2BASE_THETA[4] = {-0.25*M_PI, -0.75*M_PI, 0.75*M_PI, 0.25*M_PI};
static constexpr double B2BASE_ALPHA[4] = {M_PI, M_PI, M_PI, M_PI};
static constexpr double B2BASE_A[4]     = {0.120, 0.120, 0.120, 0.120};
static constexpr double DH_ARM_A[5]     = {0.1395, 0.115, 0.110, 0.024, 0.068};
static constexpr double DH_ARM_ALPHA[5] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, 0.0};

// ===== RT Scheduling & CPU IDs =====
static constexpr int MAIN_PRIOR = 90;
static constexpr int T265_PRIOR = 82;
static constexpr int OPTI_PRIOR = 78;
static constexpr int DXL_PRIOR  = 74;
static constexpr int SBUS_PRIOR = 70;
static constexpr int MPC_PRIOR  = 60;

static constexpr int CPU_MAIN = 11;
static constexpr int CPU_MPC  = 10;
static constexpr int CPU_T265 = 9;
static constexpr int CPU_OPTI = 9;
static constexpr int CPU_DXL  = 21;
static constexpr int CPU_SBUS = 21;

// ===== Sensor & Actuator hardcoded config =====
static constexpr const char* SBUS_PORT_NAME = "/dev/ttyUSB0";
static constexpr const char* DXL_PORT_NAME  = "/dev/ttyUSB1";
static constexpr const char* CAN_PORT_NAME  = "can0";
static constexpr const char* MOCAP_TYPE     = "optitrack";
static constexpr const char* OPTI_IP        = "192.168.10.115";
static constexpr const char* RIGIDBODY_NAME = "strider";
static constexpr const char* Log_File_NAME  = "0207_.bin";

} // namespace param

#endif // PARAMS_H