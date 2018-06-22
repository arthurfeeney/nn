
#ifndef INIT_AUX_HPP
#define INIT_AUX_HPP

#include <random>
#include "../aux.hpp"

namespace init {
namespace rand {
// Just used to simplify use when generating random numbers.

template<typename RealType = double>
class NormalRand {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<RealType> d;
public:
    NormalRand(RealType mean = 0.0, RealType var = 1.0):
        rd(), gen(rd()), d(mean, var) {}

    RealType operator()() {
        return d(gen);
    }
};


template<typename RealType = double>
class UniformRand {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<RealType> d;
public:
    UniformRand(RealType a = 0.0, RealType b = 1.0):
        gen(rd()), d(a, b) {}

    RealType operator()() {
        return d(gen);
    }
};


template<typename T, typename Generator>
void fill_1d(T vector, Generator& g) {
    for(size_t i = 0; i < vector.size(); ++i) {
        vector[i] = g();
    }
}

template<typename T, typename Generator>
void fill_2d(T matrix, Generator& g) {
    for(size_t i = 0; i < matrix.size(); ++i) {
        fill_1d(matrix[i], g);
    }
}

template<typename T, typename Generator>
void fill_3d(T& image, Generator& g) {
    for(size_t i = 0; i < image.size(); ++i) {
        fill_2d(image[i], g);
    }
}

template<typename T, typename Generator>
void fill_4d(T& input, Generator& g) {
    for(size_t i = 0; i < input.size(); ++i) {
        fill_3d(input[i], g);
    }
}

template<typename T, typename Generator>
void fill_nd(T& input, size_t rank, Generator& g) {
    //size_t rank = aux::type_rank<T>::value;     
    
    if(rank == 2) {
        fill_2d(input, g);
    }
}

} // rand
} // init

#endif
