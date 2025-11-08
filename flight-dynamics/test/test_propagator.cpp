#include <Eigen/Dense>

#include "../constants.hpp"
#include "../Propagator.cpp"

int main() {

    Propagator prop;

    prop.initial_state = (Eigen::VectorXd(7) << Constants::R_EARTH + 300.0, 0.0, 0.0, 0.0, 7.0, 0.0, 300.0).finished();
    prop.delta_t = 10;
    prop.Isp = 1;
    prop.t_final = 3600;

    prop.propagate();

}