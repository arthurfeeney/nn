#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include "layers.hpp"
#include "aux.hpp"

#ifndef DROPOUT_HPP
#define DROPOUT_HPP

/*
 * Have to modify during the test phase. Don't forget...
 */

template<typename Weight>
class Dropout2d : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;

    Dropout2d(double keep_prob): 
        Layer_2D<Weight>("dense"),
        keep_prob(keep_prob)
    {} 

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        if(!this->is_training) {
            this->last_output = input;
            return input; // dropout does nothing during test phase!
        }
        Matrix dropped(input.size(), std::vector<Weight>(input[0].size(), 0));
        Matrix mask(input.size(), std::vector<Weight>(input[0].size(), 0));
        // generate mask of bernoulli variables. Scale by the keep_prob.
        for(int row = 0; row < mask.size(); ++row) {
            std::fill(mask[row].begin(), mask[row].end(), 
                      static_cast<Weight>(keep()) / keep_prob);
        }
        for(int row = 0; row < input.size(); ++row) {
            for(int col = 0; col < input[0].size(); ++col) {
                dropped[row][col] = input[row][col] * mask[row][col];
            }
        }
        // save mask because it is used in backprop, not dropped. 
        // even though dropped is the actual output of the layer.
        this->last_output = mask; 
        return dropped;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size(), 0)); 
        for(int i = 0; i < d_out.size(); ++i) {
            for(int j = 0; j < d_out[0].size(); ++j) {
                d_input[i][j] = d_out[i][j] * this->last_output[i][j];
            }
        }
        return d_input;
    }

private:
    double keep_prob;
    
    bool keep() {
        return aux::gen_double(0, 1) <= keep_prob;
    }
};

#endif
