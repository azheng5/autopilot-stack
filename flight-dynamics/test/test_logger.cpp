#include <Eigen/Dense>

#include "../Logger.cpp"

int main() {

    Logger log(3, 3, "a,b,c");
    log.data = Eigen::MatrixXd::Ones(3,3);
    log.save_to_csv("test_data.csv");

}