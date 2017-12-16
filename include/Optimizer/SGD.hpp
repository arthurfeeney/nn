
#ifndef SGD_HPP
#define SGD_HPP

#include "../aux.hpp"

/*
 * Stochastic Gradient Descent for updating networks.
 */

struct SGD {
    SGD() {}

    template<typename Matrix>
    void perform(Matrix& weights, const Matrix& dw, double learning_rate) {
        auto scale_dw = aux::scale_mat(dw, -learning_rate);
        for(size_t row = 0; row < dw.size(); ++row) {
            for(size_t col = 0; col < dw[0].size(); ++col) {
                weights[row][col] += scale_dw[row][col];
            }
        }
    }
};

#endif
