
#ifndef MOMENTUM_HPP
#define MOMENTUM_HPP

#include <vector>
#include "../aux.hpp"

template<int Num, int Denom, typename Weight = double>
class Momentum {
private:
    std::vector<std::vector<Weight>> v; // v is the 'velocity'
    double momentum = static_cast<double>(Num) / static_cast<double>(Denom);

    bool v_initialized = false;

    void initialize_v(size_t height, size_t width) {
        v.resize(height, std::vector<Weight>(width, 0.0));
    }

public:
    Momentum() {}
    
    template<typename Matrix>
    void perform(Matrix& weights, const Matrix& dw, double learning_rate) {
        if(!v_initialized) {
            initialize_v(dw.size(), dw[0].size());
            v_initialized = true;
        }
        v = aux::matadd(aux::scale_mat(v, momentum), 
                        aux::scale_mat(dw, -learning_rate));
        for(size_t row = 0; row < dw.size(); ++row) {
            for(size_t col = 0; col < dw[0].size(); ++col) {
                weights[row][col] += v[row][col];
            }
        }
    }
};

#endif
