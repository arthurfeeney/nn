
#ifndef UNIFORM_HPP
#define UNIFORM_HPP

#include "../aux.hpp"
#include "init_aux.hpp"

namespace init {

template<typename T, typename RealType = double>
void uniform(T& input, size_t rank, RealType a, RealType b) {
    /*
     * Modifies input in-place. uniform random values in [a, b).
     */
    rand::UniformRand<RealType> u(a, b);

    rand::fill_nd<T, rand::UniformRand<RealType>>(input, rank, u);
}

}


#endif
