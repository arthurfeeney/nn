
#ifndef SGD_HPP
#define SGD_HPP

#include "aux.hpp"

/*
 * Stochastic Gradient Descent for updating networks.
 */

struct SGD {
    SGD() {}
    SGD(size_t height, size_t width) {}

    template<typename Matrix>
    Matrix perform(const Matrix& dw, double learning_rate) {
        return aux::scale_mat(dw, -learning_rate);
    }
};

#endif
