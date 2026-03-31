#ifndef PARAMS_H
#define PARAMS_H

#include <chrono>
#include <Eigen/Dense>

static inline constexpr double inv_sqrt2 = 0.7071067811865474617150084668537601828575;  // 1/sqrt(2)

/* ALL TUNABLE & CONFIGUABLE PARAMETERS ARE HERE */

namespace param {

enum class PathStage : uint8_t {
  HOLD_LEFT   = 0,
  MOVE_L2R    = 1,
  HOLD_RIGHT  = 2,
  MOVE_R2L    = 3,
};

// ===== Geometry position control gain =====
static constexpr double kX[3]  = {50.0, 50.0, 46.0};    // Position gain [x, y, z]
static constexpr double kV[3]  = {19.0, 19.0, 32.0};    // Velocity gain [x, y, z]
static constexpr double kIX[3] = {15.0, 15.0, 30.0};     // Integral gain [x, y, z]

// ===== Geometry attitude control gain =====
static constexpr double kR[3]  = {50.0, 50.0,  14.0}; // Rotational gain [roll, pitch, yaw]
static constexpr double kW[3]  = {11.0, 11.0,  5.50}; // angular Velocity gain [roll, pitch, yaw]

// ===== UAV Parameter =====
static constexpr double Jx_bar = 0.138;  // Nominal : bar 0.068 + Mass 0.07 = 0.138 [kg m^2] 
static constexpr double Jz_bar = 0.138;  // Nominal : bar 0.068 + Mass 0.07 = 0.138 [kg m^2]
static constexpr double M_bar  = 1.0;   // Nominal : bar 0.460 + Mass 0.47    = 0.93 [Kg]

static constexpr double J[9] = {    0.3 + Jx_bar, -0.0006,          -0.0006,
                                         -0.0006,     0.3,           0.0006,
                                         -0.0006,  0.0006,  0.5318 + Jz_bar}; // [kg m^2]

static constexpr double M  = 7.40 + M_bar;    // [kg]
static constexpr double G  = 9.80665;         // [m/s^2] (must be positive)

inline constexpr double COM_OFF_X   = +0.00; // distance from the body frame to com position (x) [m]
inline constexpr double COM_OFF_Y   = -0.03; // distance from the body frame to com position (y) [m]

static constexpr double SATURATION_THRUST  = 40; // (85%) Maximum thrust per each propeller [N]

// ===== Control Allocation =====
static constexpr double M_link[5] = {0.374106, 0.13658, 0.0415148, 0.102003, 0.3734}; //each link mass [kg]
static constexpr double M_body = 1.6845345+1.0;   // center body + load mass [kg]

static constexpr double SERVO_DELAY_ALPHA = 0.093158;  // yaw trimming
static constexpr double SERVO_DELAY_BETA  = 1.0 - SERVO_DELAY_ALPHA; // this not tunable

// ===== Filter cutoff frequencys =====
static constexpr double GYRO_XY_CUTOFF_HZ   = 30.0;
static constexpr double GYRO_Z_CUTOFF_HZ    =  5.0;
static constexpr double ALPHA_LPF_CUTOFF_HZ =  5.0; // Not use in Controller (Only plot)
static constexpr double OPTI_VEL_CUTOFF_HZ  =  4.0;
static constexpr double ACC_LPF_CUTOFF_HZ   =  5.0; // Not use in Controller (Only plot)

// ===== Various saturation parameters =====
static constexpr double EX_NORM_MAX         = 2.0;  // position control, position error max [m]
static constexpr double kIX_SAT[3]          = {15.0, 15.0, 50.0}; // position control, integral max on x,y,z [N]

static constexpr double ROLL_TORQUE_SAT     = 11.0;  // attitude control, torque max [Nm]
static constexpr double PITCH_TORQUE_SAT    = 11.0;  // attitude control, torque max [Nm]
static constexpr double YAW_TORQUE_SAT      = 5.0;  // attitude control, torque max [Nm]
static constexpr double ER_NORM_MAX         = 50.0 * M_PI / 180.0; // attitude control, attitude error max [rad]

static constexpr double MINIMUM_THRUST_SAT  = 8.7;  // sequential control allocation, thrust of each prop (min 5%) [N]
static constexpr double MAXIMUM_THRUST_SAT  = 53.3; // sequential control allocation, thrust of each prop (max 98%) [N]
static constexpr double REACTION_TORQUE_SAT = 5.0;  // sequential control allocation, reaction torque max [Nm]
static constexpr double TILT_ANGLE_SAT      = 25.0 * M_PI / 180.0; // sequential control allocation, tilt angle max [rad]

// ===== SBUS command =====
static constexpr double SBUS_X_RANGE       = 1.0;   // [m] k mapped to [-k, +k]
static constexpr double SBUS_Y_RANGE       = 1.0;   // [m] k mapped to [-k, +k]
static constexpr double SBUS_Z_RANGE       = 1.3;   // [m] k mapped to [ 0, -k]
static constexpr double SBUS_YAW_SPEED     = 20.0;  // [deg/s] @60Hz SBUS rate
static constexpr double SBUS_COTXY_RANGE[2] = {-0.04, 0.04};  // [m]

// ===== OptiTrack offsets =====
static constexpr double OPTI_X_OFFSET  = -0.380; // [m] Opti perspective coordinate system
static constexpr double OPTI_Y_OFFSET  = +0.430; // [m] Opti perspective coordinate system

// ===== Control Frequencies =====
static constexpr std::chrono::steady_clock::duration CTRL_DT       = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(2700)); // ~370Hz
static constexpr std::chrono::steady_clock::duration MAX_PULL_TICK = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(800));
static constexpr std::chrono::steady_clock::duration MPC_DT        = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::microseconds(5000)); // 200Hz

// ===== MPC parameters  =====
inline constexpr double      MPC_STEP_DT = 1.0 / 100.0; // This value must be same as >> DT << on params.py
inline constexpr std::size_t N_STEPS_REQ = 100; // This value must be less than >> N << on params.py
inline constexpr std::size_t MPC_NX      = 22;  // This value must be same as >> self.yes_cot_nx << on solver.py
inline constexpr std::size_t MPC_NU      = 11;  // This value must be same as >> self.yes_cot_nu << on solver.py
inline constexpr std::size_t MPC_NP      = 25;  // This value must be same as >> self.yes_cot_np << on solver.py
inline constexpr std::chrono::steady_clock::duration MPC_TIMEOUT_DURATUION = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(static_cast<double>(N_STEPS_REQ-1) * MPC_STEP_DT));

inline const     Eigen::Vector3d r1_init       = Eigen::Vector3d( 0.24, -0.24, -0.24); // rotor-1 inital position
inline const     Eigen::Vector3d r2_init       = Eigen::Vector3d(-0.24, -0.24, -0.24); // rotor-2 inital position
inline const     Eigen::Vector3d r3_init       = Eigen::Vector3d(-0.24,  0.24, -0.24); // rotor-3 inital position
inline const     Eigen::Vector3d r4_init       = Eigen::Vector3d( 0.24,  0.24, -0.24); // rotor-4 inital position

inline constexpr double MPC_OFF_TIME_CONSTANT = 0.8; // [sec] each arm goes to initial position when MPC-off or Solve-failed
inline const     double GOES_2_ZERO_A         = std::exp(-std::chrono::duration_cast<std::chrono::duration<double>>(CTRL_DT).count() / MPC_OFF_TIME_CONSTANT); // not a tunable parameter
inline const     double GOES_2_ZERO_B         = 1.0 - GOES_2_ZERO_A;                        // not a tunable parameter

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

// ===== Workspace constraint =====
inline constexpr double MAX_STRETCH       = 0.2925; // Maximum distance arm can extend from the base [m]
inline constexpr double MIN_STRETCH       = 0.1506; // Minimum distance arm can extend from the base [m]
inline constexpr double ROTOR_DIAMETER    = 0.44;   // propeller diameter [m]

inline constexpr double STRETCH_FAIL_MARGIN    = 0.2; // [m]
inline constexpr double COLLISION_FAIL_MARGIN  = 0.2; // [m]
inline constexpr double GUARD_MOVE_MARGIN      = 0.2; // [m]
inline constexpr double ALPHA_MIN[4] = {-105.0 * M_PI/180.0, -195.0 * M_PI/180.0,  75.0 * M_PI/180.0, -15.0 * M_PI/180.0};
inline constexpr double ALPHA_MAX[4] = {  15.0 * M_PI/180.0,  -75.0 * M_PI/180.0, 195.0 * M_PI/180.0, 105.0 * M_PI/180.0};

// ===== Arm parameters =====
static constexpr double B2BASE_THETA[4] = {-0.25*M_PI, -0.75*M_PI, 0.75*M_PI, 0.25*M_PI};
static constexpr double B2BASE_ALPHA[4] = {M_PI, M_PI, M_PI, M_PI};
static constexpr double B2BASE_A[4]     = {0.120, 0.120, 0.120, 0.120};
inline constexpr double B2BASE_X[4]     = { 0.12*inv_sqrt2, -0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2}; // x-distance from the body frame to each base frame [m]
inline constexpr double B2BASE_Y[4]     = {-0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2,  0.12*inv_sqrt2}; // y-distance from the body frame to each base frame [m]
static constexpr double DH_ARM_A[5]     = {0.1395, 0.115, 0.110, 0.024, 0.068};
static constexpr double DH_ARM_ALPHA[5] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, 0.0};
static constexpr double D_LINK[5] = {0.0995, 0.0840, 0.0550, 0.0120, 0.0480}; // link CoM distance [m]

// ===== Path planning Parameters =====
static inline const Eigen::Vector3d DEFAULT_pos       = Eigen::Vector3d(-OPTI_X_OFFSET, -1.25, -1.3);
static inline const Eigen::Vector3d Pos_L             = Eigen::Vector3d(-OPTI_X_OFFSET, -1.25, -1.3);
static inline const Eigen::Vector3d Pos_R             = Eigen::Vector3d(-OPTI_X_OFFSET, +1.25, -1.3);

static constexpr double DEFAULT_POS_TOL             = 0.01;  // [m]
static constexpr double DEFAULT_ARM_TOL             = 0.005; // [m]
static constexpr double PATH_T_MOVE                 = 1.5;
static constexpr double PATH_SETTLE_MAX             = 1.5;

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

} // namespace param

#endif // PARAMS_H