#ifndef GYRO_EKF_H
#define GYRO_EKF_H

#include "params.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

class GyroEKF {
 public:
  GyroEKF() {
    dt_ = 1.0/240.0;

    xhat_.setZero(); // [phi, theta, psi, p, q, r]

    P_ = 1e-2 * Eigen::Matrix<double, 6, 6>::Identity();

    Q_.setIdentity(); // Process Noise
    Q_.diagonal() << 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 2e-5;

    R_.setIdentity(); // Measurement Noise
    R_.diagonal() << 1e-8, 1e-8, 1e-7, 1e-4, 1e-4, 5e-4; // 5e-4 down // 1e-6 up // 5e-5 up //

    J_ << param::J[0], param::J[1], param::J[2], 
          param::J[3], param::J[4], param::J[5], 
          param::J[6], param::J[7], param::J[8];

    J_inv_ = J_.inverse();
  }

  Eigen::Vector3d step(const Eigen::Vector3d& tau, const Eigen::Vector3d& angle_meas, const Eigen::Vector3d& gyro_meas) {
    Eigen::Matrix<double, 6, 1> y_meas;
    y_meas << angle_meas, gyro_meas;

    predict(tau);
    update(y_meas);

    return xhat_.segment<3>(3);
  }

 private:
  double dt_;
  Eigen::Matrix<double, 6, 1> xhat_;
  Eigen::Matrix<double, 6, 6> P_;
  Eigen::Matrix<double, 6, 6> Q_;
  Eigen::Matrix<double, 6, 6> R_;
  Eigen::Matrix3d J_;
  Eigen::Matrix3d J_inv_;

  static double wrapPi(double a) {
    while (a > M_PI) {a -= 2.0 * M_PI;}
    while (a <= -M_PI) {a += 2.0 * M_PI;}
    return a;
  }

  static void wrapEulerInState(Eigen::Matrix<double, 6, 1>& x) {
    x(0) = wrapPi(x(0));
    x(1) = wrapPi(x(1));
    x(2) = wrapPi(x(2));
  }

  static Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d M;
    M << 0.0, -v(2),  v(1),
         v(2),  0.0, -v(0),
        -v(1),  v(0),  0.0;
    return M;
  }

  static double safeCos(double a) {
    constexpr double eps = 1e-6;
    const double c = std::cos(a);
    if (std::abs(c) < eps) {return (c >= 0.0) ? eps : -eps;}
    return c;
  }

  Eigen::Matrix3d calcL(const Eigen::Vector3d& eul) const {
    const double phi = eul(0);
    const double th  = eul(1);

    const double sphi = std::sin(phi);
    const double cphi = std::cos(phi);
    const double cth  = safeCos(th);
    const double tth  = std::sin(th) / cth;
    const double secth = 1.0 / cth;

    Eigen::Matrix3d L;
    L << 1.0, sphi * tth, cphi * tth,
         0.0, cphi,      -sphi,
         0.0, sphi * secth, cphi * secth;
    return L;
  }

  Eigen::Matrix3d calc_dL_dphi(const Eigen::Vector3d& eul) const {
    const double phi = eul(0);
    const double th  = eul(1);

    const double sphi = std::sin(phi);
    const double cphi = std::cos(phi);
    const double cth  = safeCos(th);
    const double tth  = std::sin(th) / cth;
    const double secth = 1.0 / cth;

    Eigen::Matrix3d M;
    M << 0.0, cphi * tth, -sphi * tth,
         0.0, -sphi,      -cphi,
         0.0, cphi * secth, -sphi * secth;
    return M;
  }

  Eigen::Matrix3d calc_dL_dtheta(const Eigen::Vector3d& eul) const {
    const double phi = eul(0);
    const double th  = eul(1);

    const double sphi = std::sin(phi);
    const double cphi = std::cos(phi);
    const double cth  = safeCos(th);
    const double secth = 1.0 / cth;
    const double sec2th = secth * secth;
    const double tth = std::sin(th) / cth;

    Eigen::Matrix3d M;
    M << 0.0, sphi * sec2th, cphi * sec2th,
         0.0, 0.0,           0.0,
         0.0, sphi * secth * tth, cphi * secth * tth;
    return M;
  }

  Eigen::Matrix3d calcATheta(const Eigen::Vector3d& eul, const Eigen::Vector3d& omg) const {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    A.col(0) = calc_dL_dphi(eul) * omg;
    A.col(1) = calc_dL_dtheta(eul) * omg;
    A.col(2).setZero();
    return A;
  }

  Eigen::Vector3d omegaDot(const Eigen::Vector3d& omg, const Eigen::Vector3d& tau) const {
    return J_inv_ * (tau - omg.cross(J_ * omg));
  }

  Eigen::Matrix3d calcAOmega(const Eigen::Vector3d& omg) const {
    // General expression:
    // A_omega = d/domega [ J^{-1}(tau - omega x (J omega)) ]
    //         = J^{-1}[J omega]_x - J^{-1}[omega]_x J
    return J_inv_ * hat(J_ * omg) - J_inv_ * hat(omg) * J_;
  }
  
  void predict(const Eigen::Vector3d& tau) {
    const Eigen::Vector3d eul = xhat_.segment<3>(0);
    const Eigen::Vector3d omg = xhat_.segment<3>(3);

    const Eigen::Matrix3d L = calcL(eul);
    const Eigen::Vector3d eul_dot = L * omg;
    const Eigen::Vector3d omg_dot = omegaDot(omg, tau);

    Eigen::Matrix<double, 6, 1> x_pred;
    x_pred.segment<3>(0) = eul + dt_ * eul_dot;
    x_pred.segment<3>(3) = omg + dt_ * omg_dot;
    wrapEulerInState(x_pred);

    const Eigen::Matrix3d A_theta = calcATheta(eul, omg);
    const Eigen::Matrix3d A_omega = calcAOmega(omg);

    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
    F.block<3, 3>(0, 0) += dt_ * A_theta;
    F.block<3, 3>(0, 3)  = dt_ * L;
    F.block<3, 3>(3, 3) += dt_ * A_omega;

    xhat_ = x_pred;
    P_ = F * P_ * F.transpose() + Q_;

    P_ = 0.5 * (P_ + P_.transpose());
  }

  void update(const Eigen::Matrix<double, 6, 1>& y_meas) {
    // Measurement model: y = Hx + v, H = I
    static const Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();

    Eigen::Matrix<double, 6, 1> innov = y_meas - xhat_;

    // Wrap angle residuals
    innov(0) = wrapPi(innov(0));
    innov(1) = wrapPi(innov(1));
    innov(2) = wrapPi(innov(2));

    const Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + R_;
    const Eigen::Matrix<double, 6, 6> K = P_ * H.transpose() * S.inverse();

    xhat_ = xhat_ + K * innov;
    wrapEulerInState(xhat_);

    const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
    P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();

    P_ = 0.5 * (P_ + P_.transpose());
  }
};

#endif // GYRO_EKF_H