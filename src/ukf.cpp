#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

#define EPSILON 0.001

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
    // initial state vector
    x_ = Eigen::VectorXd(5);

    // initial covariance matrix
    P_ = Eigen::MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.8;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.4;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_x_;
    n_sig_ = 2 * n_aug_ + 1;
    Xsig_pred_ = Eigen::MatrixXd(n_x_, n_sig_);
    weights_ = Eigen::VectorXd(n_sig_);
    time_us_ = 0.0;
    is_initialized_ = false;
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {
        x_ << 1, 1, 1, 1, 0.1;
        P_ << 0.15, 0, 0, 0, 0,
            0, 0.15, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            x_(0) = meas_package.raw_measurements_(0);
            x_(1) = meas_package.raw_measurements_(1);
        }
        else {
            double ro = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            x_(0) = ro * std::cos(phi);
            x_(1) = ro * std::sin(phi);
        }

        is_initialized_ = true;
        time_us_ = meas_package.timestamp_;
        return;
    }

    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    Prediction(dt);

    int n_z = (int)meas_package.sensor_type_;
    UpdateLaserOrRadar(meas_package, n_z);
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
    // Generate sigma points
    lambda_ = 3 - n_x_;
    Eigen::MatrixXd Xsig = Eigen::MatrixXd(n_x_, 2 * n_x_ + 1);
    Eigen::MatrixXd A = P_.llt().matrixL();
    Xsig.col(0) = x_;

    for (int i = 0; i < n_x_; i++) {
        Xsig.col(i + 1) = x_ + std::sqrt(lambda_ + n_x_) * A.col(i);
        Xsig.col(i + 1 + n_x_) = x_ - std::sqrt(lambda_ + n_x_) * A.col(i);
    }

    // Augment sigma points
    lambda_ = 3 - n_aug_;
    Eigen::VectorXd x_aug = Eigen::VectorXd(n_aug_);
    Eigen::MatrixXd P_aug = Eigen::MatrixXd(n_aug_, n_aug_);
    Eigen::MatrixXd Xsig_aug = Eigen::MatrixXd(n_aug_, n_sig_);

    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    Eigen::MatrixXd L = P_aug.llt().matrixL();

    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + std::sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - std::sqrt(lambda_ + n_aug_) * L.col(i);
    }

    // Predict sigma points
    for (uint i = 0; i < n_sig_; i++) {
        double px_p, py_p;

        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        if (std::fabs(yawd) > EPSILON) {
            px_p = p_x + v / yawd * (std::sin(yaw + yawd * delta_t) - std::sin(yaw));
            py_p = p_y + v / yawd * (std::cos(yaw) - std::cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * std::cos(yaw);
            py_p = p_y + v * delta_t * std::sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * std::cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * std::sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    // Convert sigma points
    double weight_0 = lambda_ / (lambda_ + n_aug_);
    weights_(0) = weight_0;
    for (uint i = 1; i < n_sig_; i++) {
        double weight = 0.5 / (n_aug_ + lambda_);
        weights_(i) = weight;
    }

    x_.fill(0.0);
    for (uint i = 0; i < n_sig_; i++) {
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    P_.fill(0.0);
    for (uint i = 0; i < n_sig_; i++) {
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

void UKF::UpdateLaserOrRadar(MeasurementPackage meas_package, int n_z) {
    Eigen::VectorXd z = meas_package.raw_measurements_;
    Eigen::MatrixXd Zsig = Eigen::MatrixXd(n_z, n_sig_);
    Eigen::MatrixXd R = Eigen::MatrixXd(n_z, n_z);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        for (uint i = 0; i < n_sig_; i++) {
            double p_x = Xsig_pred_(0, i);
            double p_y = Xsig_pred_(1, i);
            Zsig(0, i) = p_x;
            Zsig(1, i) = p_y;
        }
    }
    else {
        for (uint i = 0; i < n_sig_; i++) {
            double p_x = Xsig_pred_(0, i);
            double p_y = Xsig_pred_(1, i);
            double v = Xsig_pred_(2, i);
            double yaw = Xsig_pred_(3, i);

            double v1 = std::cos(yaw) * v;
            double v2 = std::sin(yaw) * v;

            Zsig(0, i) = std::sqrt(p_x * p_x + p_y * p_y);
            Zsig(1, i) = std::atan2(p_y, p_x);
            Zsig(2, i) = (p_x * v1 + p_y * v2) / std::sqrt(p_x * p_x + p_y * p_y);
        }
    }

    Eigen::VectorXd z_pred = Eigen::VectorXd(n_z);
    z_pred.fill(0.0);
    for (uint i = 0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    Eigen::MatrixXd S = Eigen::MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (uint i = 0; i < n_sig_; i++) {
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        R << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
    }
    else {
        R << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;
    }

    S = S + R;

    Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (uint i = 0; i < n_sig_; i++) {
        Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
        Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    Eigen::MatrixXd K = Tc * S.inverse();
    Eigen::VectorXd z_diff = z - z_pred;

    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
}
