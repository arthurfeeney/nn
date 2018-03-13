
#ifndef HASH_FUNCTOR_HPP
#define HASH_FUNCTOR_HPP

#include <vector>

#include "aux.hpp"

template<typename Weight>
class HashFunctor {
private:
    std::vector<Weight> a;
    Weight r;
    Weight b; // in [0.0, r]

    size_t table_size;

public:
    HashFunctor(size_t dim, size_t table_size, double r): 
        a(dim), 
        r(r), 
        b(aux::gen_double(0, r)),
        table_size(table_size)
    {
        for(auto& elem : a) {
            elem = aux::gen_double(-0.5, 0.5);
        }
    }

    HashFunctor(HashFunctor&& other):
        a(other.a),
        r(other.r),
        b(other.b),
        table_size(other.table_size)
    {}

    HashFunctor(const HashFunctor& other):
        a(other.a),
        r(other.r),
        b(other.b),
        table_size(other.table_size)
    {}
        


    //HashFunctor(HashFunctor&& other): 

    size_t operator()(const std::vector<Weight>& x) {
        Weight trans = aux::dot<std::vector<Weight>, Weight>(a, x);
        trans += b;
        return static_cast<size_t>(std::floor(trans / r)) % this->table_size;
    }

};

#endif
