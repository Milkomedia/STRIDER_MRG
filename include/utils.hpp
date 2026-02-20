#ifndef UTILS_H
#define UTILS_H

#include "params.hpp"

#include <unistd.h>

enum class Phase : uint8_t {
  READY          = 0,  // program started
  ARMED          = 1,  // all sanity checked
  IDLE           = 2,  // all propellers are idling
  RISING         = 3,  // propeller thrust increasing
  GAC_FLIGHT     = 4,  // flight with geometry controller
  MRG_NO_COT     = 5,  // flight with reference governor
  MRG_YES_COT    = 6,  // flight with reference governor (use CoT moving)
  KILLED         = 99, // killed; (It's not used as a trigger, just a state representation)
};

struct State {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();       // current linear position [m] (Optitrack)
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();       // current linear velocity [m/s] (Optitrack)
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();       // current linear acceleration [m/s^2] (Optitrack)
  Eigen::Matrix3d R   = Eigen::Matrix3d::Identity();   // current Rotation matrix [SO3] (T265)
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();     // current angular velocity [rad/s] (T265)
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero();     // current angular acceleration [rad/s^2] (T265, just logging - not used)
  Eigen::Vector3d r1  = Eigen::Vector3d::Zero();       // current rotor1 position [m] (Dynamixel)
  Eigen::Vector3d r2  = Eigen::Vector3d::Zero();       // current rotor2 position [m] (Dynamixel)
  Eigen::Vector3d r3  = Eigen::Vector3d::Zero();       // current rotor3 position [m] (Dynamixel)
  Eigen::Vector3d r4  = Eigen::Vector3d::Zero();       // current rotor4 position [m] (Dynamixel)
  Eigen::Vector3d r_cot = Eigen::Vector3d::Zero();     // current b_p_Cot position [m] (Dynamixel)
  double arm_q[20] = {0.0};                            // current joint angle [rad] (Dynamixel)
  Eigen::Vector3d r_com = Eigen::Vector3d::Zero();     // current estimated CoM position [m] 
};

struct Command {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();        // desired linear position     [m]
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();        // desired linear velocity     [m/s]
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();        // desired linear acceleration [m/s^2]
  double yaw = 0.0;                                     // desired heading yaw angle [rad] (projection of heading)
  Eigen::Vector3d heading = Eigen::Vector3d(1,0,0);     // desired heading vector [unit vector]
  // This can only be changed by Control Allocation
  double tauz_bar  = 0.0;                               // current yaw thrust torque [N.m] (Sequential control allocation)
  // These can only be changed by MRG
  Eigen::Vector3d d_theta = Eigen::Vector3d::Zero();    // desired delta theta [rad]
  Eigen::Vector3d r1 = Eigen::Vector3d::Zero();         // desired rotor1 position [m], z-element is not updated
  Eigen::Vector3d r2 = Eigen::Vector3d::Zero();         // desired rotor2 position [m], z-element is not updated
  Eigen::Vector3d r3 = Eigen::Vector3d::Zero();         // desired rotor3 position [m], z-element is not updated
  Eigen::Vector3d r4 = Eigen::Vector3d::Zero();         // desired rotor4 position [m], z-element is not updated
  // NOTE: r1234 can be manually changed by SBUS during [GAC_FLIGHT] phase.
};

struct Butter {
  // 2nd-order Butterworth states
  double bf_x1_{0.0};
  double bf_x2_{0.0};

  // cutoff related (rad/s)
  double wc_{0.0};
  double wc2_{0.0};
  double wc_sqrt2_{0.0};

  uint64_t last_ns_{0};
  bool has_last_time_{false};

  explicit Butter(const double cutoff_hz) {
    wc_ = 2.0 * M_PI * cutoff_hz;
    wc2_ = wc_ * wc_;
    wc_sqrt2_ = std::sqrt(2.0) * wc_;
  }

  inline double update(const double raw, const uint64_t now_ns) {
    // time count
    double dt = 1e12;
    if (has_last_time_) {
      if (now_ns > last_ns_) {dt = static_cast<double>(now_ns - last_ns_) * 1e-9;}
      else {return raw;}
    }
    else {has_last_time_ = true;}
    last_ns_ = now_ns;
    if (dt <= 1e-8 || dt > 1.0) {return raw;}

    // calc filter
    const double dx1 = -wc_sqrt2_ * bf_x1_ - wc2_ * bf_x2_ + raw;
    const double dx2 = bf_x1_;

    bf_x1_ += dx1 * dt;
    bf_x2_ += dx2 * dt;

    return wc2_ * bf_x2_;
  }

  inline void reset()
  {
    bf_x1_ = 0.0;
    bf_x2_ = 0.0;
    last_ns_ = 0;
    has_last_time_ = false;
  }
};

// --------- [ Math utility ] ---------
static inline constexpr double inv_sqrt2 = 0.7071067811865474617150084668537601828575;  // 1/sqrt(2)

static inline Eigen::Vector3d quat_to_RPY(const Eigen::Quaterniond q) {
  // Quaternion to Euler angle map
  double w = q.w(); double x = q.x(); double y = q.y(); double z = q.z();

  const double xx = x*x, yy = y*y, zz = z*z;
  const double xy = x*y, xz = x*z, yz = y*z;
  const double wx = w*x, wy = w*y, wz = w*z;
  
  const double phi = std::atan2(2.0*(wx + yz), 1.0 - 2.0*(xx + yy));
  double sinp = 2.0*(wy - xz);
  sinp = std::max(-1.0, std::min(1.0, sinp));
  const double th = std::asin(sinp);
  const double psi = std::atan2(2.0*(wz + xy), 1.0 - 2.0*(yy + zz));

  return Eigen::Vector3d(phi, th, psi);
}

static inline Eigen::Matrix3d quat_to_R(const double& x, const double& y, const double& z, const double& w) {
  // Quaternion to SO3 map (t265 pose-frame -> z-down frame transformation)
  const double xx = x*x, yy = y*y, zz = z*z;
  const double xy = x*y, xz = x*z, yz = y*z;
  const double wx = w*x, wy = w*y, wz = w*z;

  // Rz(90) * Rx(180) = imu->body rotation (t265 pose to strider body)
  // Rz(90) * Rx(-90) = z-down->pose(y-up) roation (strider ground to t265 ground)
  // R = Rz(90) * Rx(-90) * R_raw * Rz(90) * Rx(180)

  // pre-calculated rotation matrix
  Eigen::Matrix3d R;
  R(0,0) = - (2.0 * (yz + wx));
  R(0,1) = - (2.0 * (xz - wy));
  R(0,2) = 1.0 - 2.0 * (xx + yy);
  R(1,0) = 2.0 * (xy - wz);
  R(1,1) = 1.0 - 2.0 * (yy + zz);
  R(1,2) = - (2.0 * (xz + wy));
  R(2,0) = - (1.0 - 2.0 * (xx + zz));
  R(2,1) = - (2.0 * (xy + wz));
  R(2,2) = 2.0 * (yz - wx);
  
  return R;
}

static inline Eigen::Vector3d R_to_rpy(const Eigen::Matrix3d& R) {
  const double r11 = R(0,0), r21 = R(1,0);
  const double r31 = R(2,0), r32 = R(2,1), r33 = R(2,2);
  const double r12 = R(0,1), r22 = R(1,1);

  // th = asin(-r31)
  double sin_th = -r31;
  sin_th = std::max(-1.0, std::min(1.0, sin_th));
  const double th = std::asin(sin_th);

  // If cos(th) is near zero => Choose phi=0
  const double cth2 = 1.0 - sin_th*sin_th; // = cos(th)^2
  if (cth2 > 1e-12) {
    const double phi = std::atan2(r32, r33);
    const double psi = std::atan2(r21, r11);
    return Eigen::Vector3d(phi, th, psi);
  }
  else {
    const double phi = 0.0;
    const double psi = std::atan2(-r12, r22);
    return Eigen::Vector3d(phi, th, psi);
  }
}

static inline Eigen::Matrix3d hat(const Eigen::Vector3d& v){
  Eigen::Matrix3d V;
  V.setZero();

  V(2,1) = v(0);
  V(1,2) = -V(2, 1);
  V(0,2) = v(1);
  V(2,0) = -V(0, 2);
  V(1,0) = v(2);
  V(0,1) = -V(1, 0);

  return V;
}

static inline Eigen::Matrix3d expm_hat(const Eigen::Vector3d& w) {
  constexpr double eps = 1e-12;

  const double th2 = w.dot(w);                 // theta^2
  const double th  = std::sqrt(th2 + eps);     // theta

  const double A = std::sin(th) / th;
  const double B = (1.0 - std::cos(th)) / (th2 + eps);

  const Eigen::Matrix3d K = hat(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  return I + A * K + B * (K * K);
}

static inline Eigen::Vector3d diff(const Eigen::Vector3d& x_cur, const Eigen::Vector3d& x_prev, const uint64_t& t_cur_ns, const uint64_t& t_prev_ns) {
  constexpr uint64_t MinDtNs_ = 2500000ULL;   // Maximum 400 Hz
  constexpr uint64_t MaxDtNs_ = 20000000ULL;  // Minimum 50 Hz

  if (t_cur_ns <= t_prev_ns) return Eigen::Vector3d::Zero();

  uint64_t dt_ns = t_cur_ns - t_prev_ns;
  dt_ns = std::max(dt_ns, MinDtNs_);
  if (dt_ns > MaxDtNs_) {return Eigen::Vector3d::Zero();}
  
  const double inv_dt = 1.0 / (static_cast<double>(dt_ns) * 1e-9);
  return (x_cur - x_prev) * inv_dt;
}

static inline bool is_near(const double a, const double b, const double tol) {
  return std::abs(a - b) <= tol;
}

static inline bool is_near(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const double tol) {
  return (a - b).norm() <= tol;  
}

static inline double smooth(const double x, const double x_d, const double alpha) {
  return (1.0 - alpha) * x + alpha * x_d;
}

static inline Eigen::Vector3d smooth(const Eigen::Vector3d& x, const Eigen::Vector3d& x_d, const double alpha) {
  return (1.0 - alpha) * x + alpha * x_d;
}

// --------- [ Kinematics ] ---------
static inline void onearm_IK(const Eigen::Vector3d& pos, const Eigen::Vector3d& heading, double out5[5]) {
  Eigen::Vector3d heading_in = heading;
  const double a2_sq = param::DH_ARM_A[1]*param::DH_ARM_A[1];
  const double a3_sq = param::DH_ARM_A[2]*param::DH_ARM_A[2];
  const double a2a3_2 = 2.0 * param::DH_ARM_A[1] * param::DH_ARM_A[2];

  const double hn = heading_in.norm();
  if (hn > 1e-12) heading_in /= hn;

  const Eigen::Vector3d p04 = pos - param::DH_ARM_A[4] * heading_in;

  const double th1 = std::atan2(p04.y(), p04.x());
  const double c1 = std::cos(th1), s1 = std::sin(th1);

  const double cross_z  = p04.x() * heading_in.y() - p04.y() * heading_in.x();
  const double denom_xy = std::hypot(p04.x(), p04.y()) + 1e-12;
  double th5 = -std::acos(std::clamp(std::abs(cross_z) / denom_xy, -1.0, 1.0));
  
  if (th5 <= M_PI/2.0) th5 += M_PI/2.0;
  if (p04.x() * pos.y() - p04.y() * pos.x() > 0.0) th5 = -th5;
  
  Eigen::Vector3d heading_projected = heading_in - std::sin(th5) * Eigen::Vector3d(s1, -c1, 0.0);
  const double hp_n = heading_projected.norm();
  if (hp_n > 1e-12) heading_projected /= hp_n;

  const Eigen::Vector3d p01(param::DH_ARM_A[0] * c1, param::DH_ARM_A[0] * s1, 0.0);
  const Eigen::Vector3d p34 = param::DH_ARM_A[3] * heading_projected;
  const Eigen::Vector3d p03 = p04 - p34;
  const Eigen::Vector3d p31 = p03 - p01;

  const double r = std::hypot(p31.x(), p31.y());
  const double s = p31.z();
  double D = (r*r + s*s - (a2_sq + a3_sq)) / a2a3_2;
  D = std::clamp(D, -1.0, 1.0);
  const double th3 = std::acos(D);

  const double alpha = std::atan2(s, r);
  const double beta  = std::atan2(param::DH_ARM_A[2] * std::sin(th3), param::DH_ARM_A[1] + param::DH_ARM_A[2] * std::cos(th3));
  const double th2   = alpha - beta;

  const double th23 = th2 + th3;
  const double c23  = std::cos(th23), s23 = std::sin(th23);
  const Eigen::Vector3d x3(c1 * c23,  s1 * c23,  s23);
  const Eigen::Vector3d z3(    s1,      -c1,    0.0);

  Eigen::Vector3d x4_des = p34;
  const double x4n = x4_des.norm();
  if (x4n > 1e-12) x4_des /= x4n;

  const double c4 = std::clamp(x3.dot(x4_des), -1.0, 1.0);
  const double s4 = z3.dot(x3.cross(x4_des));
  const double th4 = std::atan2(s4, c4);

  out5[0] = th1;
  out5[1] = th2;
  out5[2] = th3;
  out5[3] = th4;
  out5[4] = th5;
}

static inline void IK(const Eigen::Vector3d& bPa1, const Eigen::Vector3d& bPa2, const Eigen::Vector3d& bPa3, const Eigen::Vector3d& bPa4, const Eigen::Vector4d& th_tvc, double q[20]) {
  const std::array<Eigen::Vector3d, 4> bodyParm{bPa1, bPa2, bPa3, bPa4};
  std::array<Eigen::Vector3d, 4> bodyE3arm;
  const double s1 = std::sin(th_tvc(0)); const double c1 = std::cos(th_tvc(0));
  const double s2 = std::sin(th_tvc(1)); const double c2 = std::cos(th_tvc(1));
  const double s3 = std::sin(th_tvc(2)); const double c3 = std::cos(th_tvc(2));
  const double s4 = std::sin(th_tvc(3)); const double c4 = std::cos(th_tvc(3));
  bodyE3arm[0] = Eigen::Vector3d( s1*inv_sqrt2,  s1*inv_sqrt2, -c1);
  bodyE3arm[1] = Eigen::Vector3d( s2*inv_sqrt2, -s2*inv_sqrt2, -c2);
  bodyE3arm[2] = Eigen::Vector3d(-s3*inv_sqrt2, -s3*inv_sqrt2, -c3);
  bodyE3arm[3] = Eigen::Vector3d(-s4*inv_sqrt2,  s4*inv_sqrt2, -c4);
  
  for (uint8_t i = 0; i < 4; ++i) {
    const double s = std::sin(param::B2BASE_THETA[i]);
    const double c = std::cos(param::B2BASE_THETA[i]);
    const double a = param::B2BASE_A[i];

    const double px = bodyParm[i].x();
    const double py = bodyParm[i].y();
    const double ezx = bodyE3arm[i].x();
    const double ezy = bodyE3arm[i].y();

    const double x_base =  c * px  + s * py - a;
    const double y_base =  s * px  - c * py;
    const double ex_base = c * ezx + s * ezy;
    const double ey_base = s * ezx - c * ezy;

    Eigen::Vector3d baseParm(x_base, y_base, -bodyParm[i].z());
    Eigen::Vector3d baseE3arm(ex_base, ey_base, -bodyE3arm[i].z());

    onearm_IK(baseParm, baseE3arm, &q[5 * i]);
  }
}

static inline Eigen::Matrix4d compute_DH(double a, double alpha, double d, double theta) {
  Eigen::Matrix4d T;
  const double c_th = cos(theta);
  const double s_th = sin(theta);
  const double c_a = cos(alpha);
  const double s_a = sin(alpha);
  
  T << c_th, -s_th * c_a,  s_th * s_a, a * c_th,
      s_th,   c_th * c_a, -c_th * s_a, a * s_th,
      0.0,         s_a,        c_a,        d,
      0.0,         0.0,        0.0,       1.0;
  return T;
}

static inline Eigen::Vector3d FK(const double q[20]) {
  // returns {b}->{cot} position and z-directional heading vector
  Eigen::Vector3d bpcot = Eigen::Vector3d::Zero();
  for (uint8_t i = 0; i < 4; ++i) {
    Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
    T_i *= compute_DH(param::B2BASE_A[i], param::B2BASE_ALPHA[i], 0.0, param::B2BASE_THETA[i]);
    for (int j = 0; j < 5; ++j) {T_i *= compute_DH(param::DH_ARM_A[j], param::DH_ARM_ALPHA[j], 0.0, q[5*i+j]);}
    bpcot += T_i.block<3, 1>(0, 3);
  }
  bpcot *= 0.25;

  return bpcot;
}

// --------- [ Control Allocation ] ---------
static inline void Sequential_Allocation(const double& thrust_d, const Eigen::Vector3d& tau_d, double& tauz_bar, const double arm_q[20], const Eigen::Vector3d& Pc, Eigen::Vector4d& C1_des, Eigen::Vector4d& C2_des) {

  double pcx = Pc(0); double pcy = Pc(1); double pcz = 0.0;
  
  // yaw wrench conversion
  tauz_bar = param::SERVO_DELAY_ALPHA*tau_d(2) + param::SERVO_DELAY_BETA*tauz_bar;
  double tauz_r = tau_d(2) - tauz_bar;
  double tauz_r_sat = std::clamp(tauz_r, -param::REACTION_TORQUE_SAT, param::REACTION_TORQUE_SAT);
  double tauz_t = tauz_bar + tauz_r - tauz_r_sat;

  // FK for each arm
  Eigen::Matrix<double, 3, 4> r_mea;   // calculated position vect of each arm [m]
  Eigen::Vector4d C2_mea;              // calculated tilted angle [rad]
  for (uint i = 0; i < 4; ++i) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T *= compute_DH(param::B2BASE_A[i], param::B2BASE_ALPHA[i], 0.0, param::B2BASE_THETA[i]);
    for (int j = 0; j < 5; ++j) {T *= compute_DH(param::DH_ARM_A[j], param::DH_ARM_ALPHA[j], 0.0, arm_q[5*i+j]);}
    r_mea.col(i) = T.block<3,1>(0,3);
    
    const Eigen::Vector3d heading = T.block<3,3>(0,0).col(0);
    C2_mea(i) = std::asin(std::clamp(heading.head<2>().cwiseAbs().sum() * inv_sqrt2, -0.5, 0.5));
  }

  double s1 = std::sin(C2_mea(0)); double s2 = std::sin(C2_mea(1)); double s3 = std::sin(C2_mea(2)); double s4 = std::sin(C2_mea(3));
  double c1 = std::cos(C2_mea(0)); double c2 = std::cos(C2_mea(1)); double c3 = std::cos(C2_mea(2)); double c4 = std::cos(C2_mea(3));

  // thrust allocation
  Eigen::Matrix4d A1;
  A1(0,0) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 0) - pcz) * s1 + (pcy - r_mea(1, 0)) * c1;
  A1(0,1) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 1) + pcz) * s2 + (pcy - r_mea(1, 1)) * c2;
  A1(0,2) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 2) + pcz) * s3 + (pcy - r_mea(1, 2)) * c3;
  A1(0,3) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 3) - pcz) * s4 + (pcy - r_mea(1, 3)) * c4;
  A1(1,0) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 0) + pcz) * s1 + (r_mea(0, 0) - pcx) * c1;
  A1(1,1) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 1) + pcz) * s2 + (r_mea(0, 1) - pcx) * c2;
  A1(1,2) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 2) - pcz) * s3 + (r_mea(0, 2) - pcx) * c3;
  A1(1,3) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 3) - pcz) * s4 + (r_mea(0, 3) - pcx) * c4;
  A1(2,0) = -param::PWM_ZETA * c1;
  A1(2,1) =  param::PWM_ZETA * c2;
  A1(2,2) = -param::PWM_ZETA * c3;
  A1(2,3) =  param::PWM_ZETA * c4;
  A1(3,0) = -c1;
  A1(3,1) = -c2;
  A1(3,2) = -c3;
  A1(3,3) = -c4;
  Eigen::Vector4d B1(tau_d(0), tau_d(1), tauz_r_sat, thrust_d);
  Eigen::FullPivLU<Eigen::Matrix4d> lu_1(A1);
  if (lu_1.isInvertible()) {C1_des = lu_1.solve(B1);}
  else {C1_des = (A1.transpose()*A1 + 1e-8*Eigen::Matrix4d::Identity()).ldlt().solve(A1.transpose()*B1);}

  // Thrust clamp
  for (uint8_t i = 0; i < 4; ++i) {C1_des(i) = std::clamp(C1_des(i), param::MINIMUM_THRUST_SAT, param::MAXIMUM_THRUST_SAT);}

  // tilt allocation
  Eigen::Matrix4d A2;
  A2(0,0) =  inv_sqrt2 * C1_des(0);
  A2(0,1) =  inv_sqrt2 * C1_des(1);
  A2(0,2) = -inv_sqrt2 * C1_des(2);
  A2(0,3) = -inv_sqrt2 * C1_des(3);
  A2(1,0) =  inv_sqrt2 * C1_des(0);
  A2(1,1) = -inv_sqrt2 * C1_des(1);
  A2(1,2) = -inv_sqrt2 * C1_des(2);
  A2(1,3) =  inv_sqrt2 * C1_des(3);
  A2(2,0) = inv_sqrt2 * (-pcx + pcy + r_mea(0, 0) - r_mea(1, 0)) * C1_des(0);
  A2(2,1) = inv_sqrt2 * ( pcx + pcy - r_mea(0, 1) - r_mea(1, 1)) * C1_des(1);
  A2(2,2) = inv_sqrt2 * ( pcx - pcy - r_mea(0, 2) + r_mea(1, 2)) * C1_des(2);
  A2(2,3) = inv_sqrt2 * (-pcx - pcy + r_mea(0, 3) + r_mea(1, 3)) * C1_des(3);
  A2(3,0) = inv_sqrt2 * ( r_mea(0, 0) - r_mea(1, 0)) * C1_des(0);
  A2(3,1) = inv_sqrt2 * ( r_mea(0, 1) + r_mea(1, 1)) * C1_des(1);
  A2(3,2) = inv_sqrt2 * (-r_mea(0, 2) + r_mea(1, 2)) * C1_des(2);
  A2(3,3) = inv_sqrt2 * (-r_mea(0, 3) - r_mea(1, 3)) * C1_des(3);
  Eigen::Vector4d B2(0.0, 0.0, tauz_t, 0.0);
  Eigen::FullPivLU<Eigen::Matrix4d> lu_2(A2);
  if (lu_2.isInvertible()) {C2_des = lu_2.solve(B2);}
  else {C2_des = (A2.transpose()*A2 + 1e-8*Eigen::Matrix4d::Identity()).ldlt().solve(A2.transpose()*B2);}

  // Tilt angle clamp
  for (uint8_t i = 0; i < 4; ++i) {C2_des(i) = std::clamp(C2_des(i), -param::TILT_ANGLE_SAT, param::TILT_ANGLE_SAT);}

}

static inline Eigen::Vector3d Wrench_2_Torque(const Eigen::Vector4d& F1234, const Eigen::Vector3d& r1, const Eigen::Vector3d& r2, const Eigen::Vector3d& r3, const Eigen::Vector3d& r4, const Eigen::Vector3d& Pc) {
  Eigen::Matrix<double, 3, 4> A;
  A(0,0) = -r1(1) + Pc(1);
  A(0,1) = -r2(1) + Pc(1);
  A(0,2) = -r3(1) + Pc(1);
  A(0,3) = -r4(1) + Pc(1);
  A(1,0) =  r1(0) - Pc(0);
  A(1,1) =  r2(0) - Pc(0);
  A(1,2) =  r3(0) - Pc(0);
  A(1,3) =  r4(0) - Pc(0);
  A(2,0) = -param::PWM_ZETA;
  A(2,1) =  param::PWM_ZETA;
  A(2,2) = -param::PWM_ZETA;
  A(2,3) =  param::PWM_ZETA;
  return A * F1234;
}

// --------- [ SBUS channel mapping ] ---------
static inline Eigen::Vector3d sbus_pos_map(const uint16_t ch0, const uint16_t ch1, const uint16_t ch2) {
  static constexpr double x_factor_ = param::SBUS_X_RANGE / 672.0;
  static constexpr double y_factor_ = param::SBUS_Y_RANGE / 672.0;
  static constexpr double z_factor_ = param::SBUS_Z_RANGE / 1344.0;

  Eigen::Vector3d out = Eigen::Vector3d::Zero();
  out(0) = static_cast<double>(1024 - ch0) * x_factor_;
  out(1) = static_cast<double>(ch1 - 1024) * y_factor_;
  out(2) = static_cast<double>(352 - ch2)  * z_factor_;

  return out;
}

static inline Eigen::Vector3d sbus_yaw_map(double& yaw_d, const uint16_t ch3) {
  static constexpr double yaw_factor_ = param::SBUS_YAW_SPEED * M_PI / 180.0 / 60.0 / 672.0;

  double delta_yaw = (ch3 < 1018 || ch3 > 1030) ? static_cast<double>(ch3 - 1024) * yaw_factor_ : 0.0;
  yaw_d += delta_yaw;

  yaw_d = fmod(yaw_d + M_PI, 2.0 * M_PI);
  if (yaw_d < 0) {yaw_d += 2.0 * M_PI;}
  yaw_d -= M_PI;

  Eigen::Vector3d out = Eigen::Vector3d(1,0,0);
  out(0) = std::cos(yaw_d);
  out(1) = std::sin(yaw_d);
  out(2) = 0.0;
  return out;
}

static inline double sbus_cotz_map(const uint16_t ch10) {
  static constexpr double cotz_factor_ = (param::SBUS_COTZ_RANGE[1] - param::SBUS_COTZ_RANGE[0]) / 1344.0;
  return param::SBUS_COTZ_RANGE[0] + static_cast<double>(ch10 - 352) * cotz_factor_;
}

static inline bool sbus_path_edge(const uint16_t ch5, bool& prev_on) {
  // | <---ON range-->   ON_THR-------------OFF_THR   <---OFF range--> |
  static constexpr uint16_t ON_THR  = 1500;
  static constexpr uint16_t OFF_THR = 900;

  bool on = prev_on;
  if (ch5 >= ON_THR) on = true;
  else if (ch5 <= OFF_THR) on = false;

  const bool edge = (on && !prev_on);
  prev_on = on;
  return edge;
}

// --------- [ PATH mapping ] ---------
static inline void path_generator_LR(const std::chrono::steady_clock::time_point& now, const State& s, Command& cmd, const bool btn_edge) {
  static param::PathStage stage = param::PathStage::HOLD_LEFT;
  static std::chrono::steady_clock::time_point move_t0;

  auto sine_pva = [&](const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const double tau, Eigen::Vector3d& p, Eigen::Vector3d& v, Eigen::Vector3d& a) {
    const Eigen::Vector3d dp = (p1 - p0);
    const double th = M_PI * tau;
    const double s_ = std::sin(th);
    const double c_ = std::cos(th);

    p = p0 + dp * (0.5 * (1.0 - c_));
    v = dp * (0.5 * (M_PI / param::PATH_T_MOVE) * s_);
    a = dp * (0.5 * (M_PI / param::PATH_T_MOVE) * (M_PI / param::PATH_T_MOVE) * c_);
  };

  switch (stage)
  {
    case param::PathStage::HOLD_LEFT:
      cmd.pos = param::P_L;
      cmd.vel.setZero();
      cmd.acc.setZero();
      if (btn_edge) { move_t0 = now; stage = param::PathStage::MOVE_L2R; }
      break;

    case param::PathStage::MOVE_L2R: {
      const double dt  = std::chrono::duration<double>(now - move_t0).count();
      const double tau = std::clamp(dt / param::PATH_T_MOVE, 0.0, 1.0);

      sine_pva(param::P_L, param::P_R, tau, cmd.pos, cmd.vel, cmd.acc);

      if (tau >= 1.0 || dt >= param::PATH_SETTLE_MAX || is_near(s.pos, param::P_R, param::DEFAULT_POS_TOL)) { stage = param::PathStage::HOLD_RIGHT; }
    } break;

    case param::PathStage::HOLD_RIGHT:
      cmd.pos = param::P_R;
      cmd.vel.setZero();
      cmd.acc.setZero();
      if (btn_edge) { move_t0 = now; stage = param::PathStage::MOVE_R2L; }
      break;

    case param::PathStage::MOVE_R2L: {
      const double dt  = std::chrono::duration<double>(now - move_t0).count();
      const double tau = std::clamp(dt / param::PATH_T_MOVE, 0.0, 1.0);

      sine_pva(param::P_R, param::P_L, tau, cmd.pos, cmd.vel, cmd.acc);

      if (tau >= 1.0 || dt >= param::PATH_SETTLE_MAX || is_near(s.pos, param::P_L, param::DEFAULT_POS_TOL)) { stage = param::PathStage::HOLD_LEFT; }
    } break;
  }

  cmd.heading = param::DEFAULT_heading;
}

// --------- [ ETC ] ---------
// Best-effort RT priority; will fail without CAP_SYS_NICE.
static inline void try_set_prior(int prio) {
  sched_param sp{};
  sp.sched_priority = prio;
  pthread_t th = pthread_self();
  const int ret = pthread_setschedparam(th, SCHED_FIFO, &sp);
  if (ret != 0) {
    std::fprintf(stderr, "[RT] pthread_setschedparam(SCHED_FIFO,%d) failed: %s\n", prio, std::strerror(ret));
    std::fflush(stderr);
  }
}

// Best-effort CPU pin; will fail without sufficient permission in some setups.
static inline void try_pin_cpu(int cpu_id) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu_id, &set);
  pthread_t th = pthread_self();
  (void)pthread_setaffinity_np(th, sizeof(cpu_set_t), &set);
}

static inline uint64_t now_steady_ns() {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count()
  );
}

#endif // UTILS_H