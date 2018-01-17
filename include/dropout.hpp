
#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <utility>
#include <memory>
#include <thread>

#include "layers.hpp"
#include "aux.hpp"
#include "thread_aux.hpp"


/*
 * Have to modify during the test phase. Don't forget...
 */

template<typename Weight>
class Dropout2d : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;

    Dropout2d(double keep_prob): 
        Layer_2D<Weight>("dropout"),
        keep_prob(keep_prob)
    {} 

    Dropout2d(Dropout2d&& other):
        Layer_2D<Weight>(std::move(other)),
        keep_prob(std::move(other.keep_prob))
    {}

    Dropout2d(const Dropout2d& other):
        Layer_2D<Weight>(other),
        keep_prob(other.keep_prob)
    {}

    Dropout2d* clone() {
        return new Dropout2d(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        if(!this->is_training) {
            this->last_output = input;
            return input; // dropout does nothing during test phase!
        }
        Matrix dropped(input.size(), std::vector<Weight>(input[0].size(), 0));
        Matrix mask(input.size(), std::vector<Weight>(input[0].size(), 0));
        // generate mask of bernoulli variables. Scale by the keep_prob.
        for(size_t row = 0; row < mask.size(); ++row) {
            for(size_t col = 0; col < mask[0].size(); ++col) {
                mask[row][col] = static_cast<Weight>(keep()) / keep_prob;
            }
        }
        for(size_t row = 0; row < input.size(); ++row) {
            for(size_t col = 0; col < input[0].size(); ++col) {
                dropped[row][col] = input[row][col] * mask[row][col];
            }
        }
        // save mask because it is used in backprop, not dropped. 
        // even though dropped is the actual output of the layer.
        this->last_output = mask; 
        return dropped;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        // same as forward_pass but splits up the indices of the outer
        // for loop when doing element wise product with the mask. 
        this->last_input = input;
        if(!this->is_training) {
            this->last_output = input;
            return input; 
        }
        Matrix dropped(input.size(), std::vector<Weight>(input[0].size(), 0));
        Matrix mask(input.size(), std::vector<Weight>(input[0].size(), 0));
        for(size_t row = 0; row < mask.size(); ++row) {
            for(size_t col = 0; col < mask[0].size(); ++col) {
                mask[row][col] = static_cast<Weight>(keep()) / keep_prob;
            }
        } 
        auto rows = thread_alg::split_indices(dropped.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(&Dropout2d::async_apply_drop,
                                 std::ref(rows[thread]),
                                 std::ref(mask),
                                 std::ref(input),
                                 std::ref(dropped));
        }
        for(auto& thread : threads) {
            thread.join();
        }
        this->last_output = mask;
        return dropped;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        // easy! just d_out times the mask. 
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size(), 0)); 
        for(size_t i = 0; i < d_out.size(); ++i) {
            for(size_t j = 0; j < d_out[0].size(); ++j) {
                d_input[i][j] = d_out[i][j] * this->last_output[i][j];
            }
        }
        return d_input;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        // the operation is so simple it may not even make sense to parallelize
        return backward_pass(d_out);
    }

private:
    double keep_prob; // the probability of keeping the node. 
    
    bool keep() {
        return aux::gen_double(0, 1) <= keep_prob; // keep if less then prob.
    }

    // splits the outer for loops indices.
    static void async_apply_drop(std::vector<int>& rows,
                                 const Matrix& mask,
                                 const Matrix& input,
                                 Matrix& to_drop) 
    {
        for(const auto& row : rows) {
            for(size_t col = 0; col < to_drop[0].size(); ++col) {
                to_drop[row][col] = input[row][col] * mask[row][col];
            }
        }
    }
};

#endif
