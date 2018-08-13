
#ifndef SUM_FUNC_HPP
#define SUM_FUNC_HPP

#include <array>

#include "../aux.hpp"

template<typename Weight = double>
class SumFunc : public Layer_2D<Weight> {
private:
    static Matrix sum;

public:
    using Matrix = std::vector<std::vector<Weight>>;
    
    SumFunc(size_t h, size_t w): 
        Layer_2D<Weight>("sum"), 
        sum(h, std::vector<Weight>(w, 0)) {}

    ~SumFunc = default;

    Matrix forward_pass(const Matrix& input) {
        sum = aux::matadd(sum, input);
        this->last_output = sum;
        return sum;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        // single threaded only
        return forward_pass(input);
    }

    
    Matrix backward_pass(const Matrix& d_out) {
        // no weighted layers, so it only compuptes d_inputs
        // This is a stupid function. One of the summands is returned.
        // The other is returned by modifying d_out.
        // It does this so that this can be used as a Layer_2D.. 
        
        // d_out is gradient of sum 
              
        // I want d_input wrt sum?
    }
}


#endif
