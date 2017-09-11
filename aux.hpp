
/*
* I probably could (should) have used the Eigen library for most of these.
* But, I really wanted to do all of this from scratch, soooo...
* Meh, they're all pretty easy to implement anyway. 
*/

#include <algorithm>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <iostream>
#ifndef AUX_HPP
#define AUX_HPP

namespace aux {
    // assumes depth of 2 i.e. vector<vector>, not vector<vector<...>...>
    template<typename Container>
    auto relu(const Container& c) -> Container {
        Container copy_container(c);
        for(auto& row: copy_container) {
            std::transform(row.begin(), row.end(), row.begin(),  
                           [](double item) {
                                return std::max(item, 0.0);
                           });
        }
        return copy_container;
    }

    template<typename Vect>
    auto dot(const Vect& v1, const Vect& v2) -> Vect {
        if(v1.size() != v2.size()) {
            throw("vectors don't have equal dimension.");
        }
        Vect dot(v1.size());
        for(int i = 0; i < v1.size(); ++i) {
            dot[i] = v1[i] * v2[i];
        }
        return dot;
    }

    template<typename Matrix>
    auto matmul(const Matrix& m1, const Matrix& m2) -> Matrix {
        if(m1[0].size() != m2.size()) {
            std::cout << "num of cols in m1 != num of rows in m2\n";
            std::cout << m1[0].size() << '\n';
            std::cout << m2.size() << '\n';
            
        }
        Matrix prod(m1.size());
        for(auto& item : prod) {
            item.resize(m2[0].size());
        }
        for(size_t row = 0; row < m1.size(); ++row) {
            for(size_t col = 0; col < m2[0].size(); ++col) {
                for(size_t inner = 0; inner < m2.size(); ++inner) {
                    prod[row][col] += m1[row][inner] * m2[inner][col];
                }
            }
        }
        return prod;
    }

    template<typename Matrix>
    auto matadd(const Matrix& m1, const Matrix& m2) -> Matrix {
        if(m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
            throw("matrix dimension not equal.");
        }
        Matrix summed(m1.size());
        for(auto& item : summed) {
            item.resize(m1[0].size());
        }
        for(size_t row = 0; row < m1.size(); ++row) {
            for(size_t col = 0; col < m1[0].size(); ++col) {
                summed[row][col] = m1[row][col] + m2[row][col];
            }
        }
        return summed;
    }

    template<typename Weight = double>
    auto exp(const std::vector<std::vector<Weight>>& m) 
        -> std::vector<std::vector<Weight>> {
        auto out(m);
        for(auto& row : out) {
            for(auto& item : row) {
                item = std::exp(item);            
            }
        }
        return out;
    }

    // 2d vector overload of log function. 
    template<typename Weight = double>
    auto log(const std::vector<std::vector<Weight>>& m) 
        -> std::vector<std::vector<Weight>> {
        auto out(m);
        for(auto& row : out) {
            for(auto& item : row) {
                item = std::log(item);            
            }
        }
        return out;
    }

    // 1d vector overload of log function.
    template<typename Weight = double>
    auto log(const std::vector<Weight>& m) 
        -> std::vector<Weight> {
        auto out(m);
        for(auto& item : out) {
            item = std::log(item);
        }
        return out;
    }

    template<typename Matrix>
    auto transpose(const Matrix& m) -> Matrix {
        Matrix trans(m[0].size(), std::vector<double>(m.size()));
        for(size_t row = 0; row < m.size(); ++row) {
            for(size_t col = 0; col < m[0].size(); ++col) {
                trans[col][row] = m[row][col];
            }
        }
        return trans;
    }

    double gen_double(double lower, double upper) {
        std::random_device(r);
        std::default_random_engine re(r());
        std::uniform_real_distribution<double> unif(lower, upper);
        return unif(re);
    }

};

#endif
