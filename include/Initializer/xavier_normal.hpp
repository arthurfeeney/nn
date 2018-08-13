

#ifndef XAVIER_NORMAL_HPP
#define XAVIER_NORMAL_HPP

namespace init {

template<typename Weights, typename Weight>
void xavier_normal(Weights& w, size_t fan_in, size_t fan_out, double gain = 1) 
{
    double denom = static_cast<double>(fan_in + fan_out);
    double std = gain * std::sqrt(2.0 / denom);
    rand::NormalRand<Weight> init(0, std);
    for(size_t i = 0; i < w.size(); ++i) {
        for(size_t j = 0; j < w[0].size(); ++j) {
            w[i][j] = init();
        }
    }
}

}

#endif
