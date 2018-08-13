

#ifndef KAIMING_NORMAL_HPP
#define KAIMING_NORMAL_HPP

#include <cmath>

#include "init_aux.hpp"

namespace init {

template<typename Weights, typename Weight>
void kaiming_normal(Weights& w, size_t fan_in, size_t fan_out,
                     double gain = 1, double a = 0)
{
    double denom = (1.0 + a*a) * static_cast<double>(fan_in);
    double inner = 2.0 / denom;
    double std = std::sqrt(inner);

    rand::NormalRand<Weight> init(0, std);

    for(size_t i = 0; i < w.size(); ++i) {
        for(size_t j = 0; j < w[i].size(); ++j) {
            w[i][j] = init();
        }
    }
}

} // init

#endif
