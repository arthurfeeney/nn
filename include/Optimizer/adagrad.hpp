
#ifndef ADAGRAD_HPP
#define ADAGRAD_HPP

#include <vector>
#include <cmath>
#include "../aux.hpp"

template<int Num = 9, int Denom = 10, typename Weight = double>
class Adagrad {
private:
    std::vector<std::vector<Weight>> cache; // v is the 'velocity'
    double eps = static_cast<double>(Num) / static_cast<double>(Denom);

    bool cache_initialized = false;

    void initialize_cache(size_t height, size_t width) {
        cache.resize(height, std::vector<Weight>(width, 0.0));
    }

public:
    Adagrad():cache() {}
    
    template<typename Matrix>
    void perform(Matrix& weights, const Matrix& dw, double learning_rate) {
        if(!cache_initialized) {
            initialize_cache(dw.size(), dw[0].size());
            cache_initialized = true;
        }
        for(size_t row = 0; row < dw.size(); ++row) {
            for(size_t col = 0; col < dw[0].size(); ++col) {
                cache[row][col] += dw[row][col] * dw[row][col];
            }
        }
        auto tmp(cache);
        for(auto& row : tmp) {
            for(auto& val : row) {
                val = std::sqrt(val);                
            }
        }
    
        auto scale_dx = aux::scale_mat(dw, -learning_rate);
    
        for(size_t row = 0; row < dw.size(); ++row) {
            for(size_t col = 0; col < dw[0].size(); ++col) {
                weights[row][col] += scale_dx[row][col] / (tmp[row][col]+eps); 
            }
        }
    }
};

#endif
