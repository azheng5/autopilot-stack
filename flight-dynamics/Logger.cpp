#include <fstream>
#include <iostream>

#include <Eigen/Dense>

class Logger {

public:
    Eigen::MatrixXd data;
    std::string headers;

    Logger(int num_rows, int num_cols, std::string headers) :
        data(Eigen::MatrixXd::Zero(num_rows, num_cols)),
        headers(headers) {}

    void save_to_csv(std::string file_name) {

        std::ofstream file(file_name);

        // Populate csv file
        file << this->headers << "\n";
        for (int i = 0; i < this->data.rows(); i++) {
            
            for (int j = 0; j < this->data.cols(); j++) {

                // Last column
                if (j == this->data.cols() - 1) {

                    // If also last row
                    if (i == this->data.rows() - 1) {
                        file << this->data(i,j);
                    } else {
                        file << this->data(i,j) << "\n";
                    }

                    
                } else {
                    file << this->data(i,j) << ",";
                }

            }
        }

        file.close();

        std::cout << "Logged data saved to " << file_name << std::endl;

    }

};