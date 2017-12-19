
#ifndef RMSPROP_HPP
#define RMSPRop_HPP

#include <vector>
#include <cmath>
#include "../aux.hpp"

template<int DecayNum = 9, int DecayDenom = 10, 
         int EpsNum = 1, int EpsDenom = 10000000,
         typename Weight = double>
class RMSProp {
private:
    std::vector<std::vector<Weight>> cache; // v is the 'velocity'
    
    double eps = static_cast<double>(EpsNum) / static_cast<double>(EpsDenom);

    double decay_rate = static_cast<double>(DecayNum) /
                        static_cast<double>(DecayDenom);

    bool cache_initialized = false;

    void initialize_cache(size_t height, size_t width) {
        cache.resize(height, std::vector<Weight>(width, 0.0));
    }

public:
    RMSProp():cache() {}
    
    template<typename Matrix>
    void perform(Matrix& weights, const Matrix& dw, double learning_rate) {
        if(!cache_initialized) {
            initialize_cache(dw.size(), dw[0].size());
            cache_initialized = true;
        }
        for(size_t row = 0; row < dw.size(); ++row) {
            for(size_t col = 0; col < dw[0].size(); ++col) {
                cache[row][col] = 
                    (decay_rate * cache[row][col]) + 
                    ((1 - decay_rate) * std::pow(dw[row][col], 2));
            }
        }
        auto tmp(cache);
        for(auto& row : tmp) {
            for(auto& val : row) {
                val = std::sqrt(val);                
            }
        }
    
        auto scale_dw = aux::scale_mat(dw, -learning_rate);
    
        for(size_t row = 0; row < dw.size(); ++row) {
            for(size_t col = 0; col < dw[0].size(); ++col) {
                weights[row][col] += scale_dw[row][col] / (tmp[row][col]+eps); 
            }
        }
    }
};

#endif
