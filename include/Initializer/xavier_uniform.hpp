

#ifndef XAVIER_UNIFORM_HPP
#define XAVIER_UNIFORM_HPP

#include "./init_aux.hpp"

namespace init {

template<typename Weights, typename Weight>
void xavier_uniform(Weights& w, size_t fan_in, size_t fan_out, 
                    double gain = 1) 
{
    double denom = static_cast<double>(fan_in + fan_out);
    double std = gain * std::sqrt(2.0 / denom); 
    double a = std::sqrt(3.0) * std;
    rand::UniformRand<Weight> init(-a, a);
    for(size_t i = 0; i < w.size(); ++i) {
        for(size_t j = 0; j < w[i].size(); ++j) {
            w[i][j] = init();
        }
    }
}


} // init

#endif
