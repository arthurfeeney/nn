
#ifndef NESTEROV_HPP
#define NESTEROV_HPP

#include <vector>
#include "aux.hpp"

/*
 * Does not work yet. 
 */


template<int num, int denom, int batch_size, typename Weight = double>
class NesterovMomentum {
private:
    std::vector<std::vector<Weight>> v; // v is the 'velocity'
    double momentum = static_cast<double>(num) / static_cast<double>(denom);

public:
    NesterovMomentum() {}
    NesterovMomentum(size_t gradient_height, size_t gradient_width):
        v(gradient_height * batch_size, 
          std::vector<Weight>(gradient_width * batch_size, 0.0))
    {}
    
    template<typename Matrix>
    Matrix perform(const Matrix& dw, double learning_rate) {
        return Matrix();/*
        v = aux::matsub(aux::scale_mat(v, momentum), 
                        aux::scale_mat(dw, learning_rate));
        return aux::matadd(aux::scale_mat(v_prev, -momentum),
                           aux::scale_mat(v, 1 + momentum));
                           */
    }
};

#endif
