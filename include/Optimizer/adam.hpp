
#ifndef ADAM_HPP
#define ADAM_HPP

#include <vector>
#include <cmath>
#include "../aux.hpp"

template<int Beta1Num = 9, int Beta1Denom = 10,
         int Beta2Num = 999, int Beta2Denom = 1000,
         int EpsNum = 1, int EpsDenom = 100000000,
         typename Weight = double>
class Adam {
private:
    std::vector<std::vector<Weight>> m;
    std::vector<std::vector<Weight>> v; // v is the 'velocity'
    
    double eps = static_cast<double>(EpsNum) / static_cast<double>(EpsDenom);

    double beta1 = static_cast<double>(Beta1Num) / 
                   static_cast<double>(Beta1Denom);

    double beta2 = static_cast<double>(Beta2Num) /
                   static_cast<double>(Beta2Denom);

    int iteration = 1;

    bool m_v_are_initialized = false;

    void initialize_m_v(size_t height, size_t width) {
        m.resize(height, std::vector<Weight>(width, 0.0));
        v.resize(height, std::vector<Weight>(width, 0.0));
    }

public:
    Adam():m(), v() {}
    
    template<typename Matrix>
    void perform(Matrix& weights, const Matrix& dw, double learning_rate) {
        
        if(!m_v_are_initialized) {
            initialize_m_v(dw.size(), dw[0].size());
            m_v_are_initialized = true;
        }


        m = aux::matadd(aux::scale_mat(m, beta1), aux::scale_mat(dw, 1-beta1));
        auto mt(m);
        for(auto& row : mt) {
            for(auto& val : row) {
                val /= 1-std::pow(beta1,iteration);
            }
        }
        
        auto dw_sqr(dw);
        for(auto& row : dw_sqr) {
            for(auto& val : row) {
                val *= val;
            }
        }
        v = aux::matadd(aux::scale_mat(v, beta2), 
                        aux::scale_mat(dw_sqr, 1-beta2));
        auto vt(v);
        for(auto& row : vt) {
            for(auto& val : row) {
                val /= 1-std::pow(beta2, iteration);    
            }
        }

        auto scale_mt = aux::scale_mat(mt, -learning_rate);

        auto vt_sqrt(vt);
        for(auto& row : vt_sqrt) {
            for(auto& val : row) {
                val = std::sqrt(val);
            }
        }

        for(size_t row = 0; row < dw.size(); ++row) {
            for(size_t col = 0; col < dw[0].size(); ++col) {
                weights[row][col] += scale_mt[row][col] / 
                                     (vt_sqrt[row][col] + eps);
            }
        }

        ++iteration;
    }
};

#endif
