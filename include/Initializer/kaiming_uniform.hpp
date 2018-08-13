

#ifndef KAIMING_UNIFORM_HPP
#define KAIMING_UNIFORM_HPP

#include <cmath>

#include "init_aux.hpp"

namespace init {

template<typename Weights, typename Weight>
void kaiming_uniform(Weights& w, size_t fan_in, size_t fan_out,
                     double gain = 1, double a = 0)
{
    double denom = (1.0 + a*a) * static_cast<double>(fan_in);
    double inner = 6.0 / denom;
    double bound = std::sqrt(inner);

    rand::UniformRand<Weight> init(-bound, bound);

    for(size_t i = 0; i < w.size(); ++i) {
        for(size_t j = 0; j < w[i].size(); ++j) {
            w[i][j] = init();
        }
    }
}

} // init

#endif
