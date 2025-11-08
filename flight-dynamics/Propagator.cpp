#include <bits/stdc++.h>
#include <cmath>
#include <functional>

#include <Eigen/Dense>

#include "constants.hpp"
#include "Propagator.hpp"

#include "Logger.cpp"

typedef std::function<Eigen::VectorXd (float, Eigen::VectorXd, Eigen::VectorXd)> t_ode;

class Propagator {
    
public:
    Eigen::VectorXd initial_state;
    float delta_t;
    float Isp;
    float t_final;

    void propagate() {
        
        std::cout << "Propagating for " << t_final << " seconds" << std::endl;
        // Initialize propagator
        bool terminate = false;
        Eigen::VectorXd x_curr = this->initial_state;
        int step_counter = 0;
        float t_curr = 0.0;
        Eigen::VectorXd u_curr = (Eigen::VectorXd(3) << 0.0, 0.0, 0.0).finished();
        Logger logger(this->t_final/this->delta_t + 1, 
                      1 + x_curr.rows() + u_curr.rows(), 
                      "t,rx,ry,rz,vx,vy,vz,m,ux,uy,uz");

        // Need to define EOM handle as a lambda object
        t_ode eom_handle = [this](float t, Eigen::VectorXd x, Eigen::VectorXd u) -> Eigen::VectorXd {
            return eom(t,x,u);
        };

        // Enter propagation loop
        while (!terminate) {

            Eigen::VectorXd x_next = rk4_step(eom_handle, t_curr, x_curr, u_curr);
            // Log data
            logger.data(step_counter, 0) = t_curr;
            logger.data.block(step_counter, 1, 1, x_curr.rows()) = x_curr.transpose();
            logger.data.block(step_counter, 1+x_curr.rows(), 1, u_curr.rows()) = u_curr.transpose();
            terminate = check_termination_conditions(t_curr);

            x_curr = x_next;
            t_curr = t_curr + this->delta_t;
            step_counter = step_counter + 1;

        }


        // Save data to csv
        logger.save_to_csv("test_prop.csv");

    }

    Eigen::VectorXd eom(float t, Eigen::VectorXd x, Eigen::VectorXd u) {

        // Extract state vector
        float rx = x(0);
        float ry = x(1);
        float rz = x(2);
        float vx = x(3);
        float vy = x(4);
        float vz = x(5);
        float m = x(6);
        Eigen::VectorXd r = (Eigen::VectorXd(3) << rx, ry, rz).finished();
        Eigen::VectorXd v = (Eigen::VectorXd(3) << vx, vy, vz).finished();

        // Extract control vector
        float T_mag = u.norm();

        // Compute 2-body gravity acceleration
        Eigen::VectorXd a_g = -Constants::EARTH_MU/(std::pow(r.norm(),3)) * r;

        // Compute J2 perturbing acceleration
        Eigen::VectorXd j2_unit_vec = (Eigen::VectorXd(3) << 
                                      (1 - 5*std::pow(rz/r.norm(),2)) * (rx/r.norm()), 
                                      (1 - 5*std::pow(rz/r.norm(),2)) * (ry/r.norm()), 
                                      (3 - 5*std::pow(rz/r.norm(),2)) * (rz/r.norm())).finished();
        Eigen::VectorXd a_j2 = -1.5 * Constants::J2 * (Constants::EARTH_MU/std::pow(r.norm(),2)) * 
                               std::pow(Constants::R_EARTH/r.norm(),2) * j2_unit_vec;

        // Compute atmospheric drag perturbing acceleration

        Eigen::VectorXd a_drag = (Eigen::VectorXd(3) << 0,0,0).finished();

        // Compute thrust acceleration
        Eigen::VectorXd a_thrust = u/m;

        // Compute mass derivative
        float m_dot = -T_mag / (this->Isp * Constants::G0);

        // Construct state derivative
        Eigen::VectorXd a_tot = a_g + a_j2 + a_drag + u;
        Eigen::VectorXd x_dot = (Eigen::VectorXd(7) <<
                                vx,
                                vy,
                                vz,
                                a_tot(0),
                                a_tot(1),
                                a_tot(2),
                                m_dot).finished();

        return x_dot;
    }

    Eigen::VectorXd rk4_step(t_ode fn, 
                             float t, 
                             Eigen::VectorXd x,
                             Eigen::VectorXd u) {

        Eigen::VectorXd k1 = fn(t, x, u);
        Eigen::VectorXd k2 = fn(t + this->delta_t/2, x + (this->delta_t/2)*k1, u);
        Eigen::VectorXd k3 = fn(t + this->delta_t/2, x + (this->delta_t/2)*k2, u);
        Eigen::VectorXd k4 = fn(t + this->delta_t, x + this->delta_t*k3, u);

        return x + (this->delta_t/6)*(k1 + 2*k2 + 2*k3 + k4);

    }

    bool check_termination_conditions(float t) {

        if (t >= this->t_final) {
            return true;
        }

        return false;

    }
        
};