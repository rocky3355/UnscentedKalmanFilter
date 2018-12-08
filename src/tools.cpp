#include <iostream>
#include "tools.h"

Eigen::VectorXd Tools::CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth) {
    Eigen::VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
      std::cout << "Invalid estimation or ground_truth data" << std::endl;
      return rmse;
    }

    for (uint i = 0; i < estimations.size(); ++i) {
      Eigen::VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array() * residual.array();
      rmse += residual;
    }

    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}
