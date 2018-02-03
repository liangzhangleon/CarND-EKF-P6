#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

    // KF Measurement update step
    VectorXd y_ = z - H_ * x_;
    MatrixXd S_ = H_ * P_ * H_.transpose() + R_;
    MatrixXd K_ = P_ * H_.transpose() * S_.inverse();
    MatrixXd I = MatrixXd::Identity(4, 4);

    // new state
    x_ = x_ + K_ * y_;
    P_ = (I - K_ * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

    // KF Measurement update step
    //recover state parameters
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    VectorXd hx(3);
    hx<< 0,0,0;
    hx(0) = std::sqrt(px * px + py * py);
    hx(1) = std::atan2(py, px);

    if(fabs(hx(0)) < 0.01)
        hx(2) = (px * vx + py * vy)/0.01;
    else
        hx(2) = (px * vx + py * vy)/hx(0);

    VectorXd y_ = z - hx;
    float const PI=3.14159265;
    if (y_(1) > PI)
        y_(1) -= 2 * PI;
    else if (y_(1) < -PI)
        y_(1) += 2 * PI;
    // In the following, H_ is Hj
    MatrixXd S_ = H_ * P_ * H_.transpose() + R_;
    MatrixXd K_ = P_ * H_.transpose() * S_.inverse();
    MatrixXd I = MatrixXd::Identity(4, 4);

    // new state
    x_ = x_ + K_ * y_;
    P_ = (I - K_ * H_) * P_;
}
