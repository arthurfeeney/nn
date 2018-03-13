
#ifndef DENSE_HPP
#define DENSE_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>
#include <cmath>

#include "aux.hpp"
#include "thread_aux.hpp"
#include "layers.hpp"

template <typename Opt, typename Weight = double>
struct Dense : public Layer_2D<Weight> {
private:
    Opt optimizer;
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Dense():Layer_2D<Weight>("dense") {}
    Dense(int num_nodes, int input_size, double learning_rate):
        Layer_2D<Weight>(num_nodes, input_size, "dense", learning_rate),
        optimizer()
    {
        // some form of uniform xavier intialization.
        double v = std::sqrt(6.0 / input_size);
        for(auto& row : this->weights) {
            for(auto& item : row) {
                item = aux::gen_double(-0.5 * v, 0.5 * v);
            }
        }
    }

    ~Dense() = default;

    Dense(Dense&& other): Layer_2D<Weight>(std::move(other)) {}

    Dense(const Dense& other): Layer_2D<Weight>(other) {}

    Dense* clone() {
        return new Dense(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        // computes output of a layer based on input and its weights.
        // input is bxK. b is batch size K is size of datum.
        this->last_input = input;
        // produces a bxN matrix. From bxK, KxN
        Matrix apply_weights = aux::matmul(input, this->weights);
        // adds bias to bxN matrix
        Matrix apply_bias = apply_weights;
        if(apply_bias.size() == 1) { // batch size is one.
            apply_bias = aux::matadd(apply_bias, this->bias);
        }
        else {
            for(size_t b = 0; b < apply_bias.size(); ++b) {
                for(size_t v = 0; v < this->bias[0].size(); ++v) {
                    apply_bias[b][v] += this->bias[0][v];
                }
            }
        }
        // saves the output of layer.
        this->last_output = apply_bias;
        return apply_bias; //returns the scores
    }

    // parallel version of forward_pass. Only difference is it
    // uses the parallel matmul implementation. In thread_aux.hpp
    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        
        this->last_input = input;
        Matrix apply_weights = thread_alg::matmul(input, this->weights,
                                                  n_threads);
        Matrix apply_bias = apply_weights;
        for(size_t b = 0; b < apply_bias.size(); ++b) {
            for(size_t v = 0; v < this->bias[0].size(); ++v) {
                apply_bias[b][v] += this->bias[0][v];
            }
        }
        this->last_output = apply_bias;
        return apply_bias;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        /*
        * updates weights and bias. Returns gradient of input.
        * requires:
        * - last_input: data input to network. 1xI
        * - dout: derivative of previous layer.
        * - weights: layers weights.KxN
        * - bias: layers bias. 1xN
        * finds gradients for weights, bias, and input to feed to next
        *  layer.
        */
        auto d_input = aux::matmul(d_out, aux::transpose(this->weights));

        auto d_weights = aux::matmul(aux::transpose(this->last_input), d_out);

        optimizer.perform(this->weights, d_weights, this->step_size);

        std::vector<Weight> d_bias(this->bias[0].size(), 0.0);
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                d_bias[col] += d_out[row][col];
            }
        }


        // bias is a matrix, d_bias is a vector (just cuz im dumb.)
        for(size_t row = 0; row < this->bias[0].size(); ++row) {
            this->bias[0][row] += -this->step_size * d_bias[row];
        }

        return d_input; // used for next layer.
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        auto d_input = thread_alg::matmul(d_out, aux::transpose(this->weights),
                                          n_threads);

        auto d_weights = thread_alg::matmul(aux::transpose(this->last_input),
                                            d_out, n_threads);

        optimizer.perform(this->weights, d_weights, this->step_size);

        std::vector<Weight> d_bias(this->bias[0].size(), 0.0);
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                d_bias[col] += d_out[row][col];
            }
        }

        for(size_t row = 0; row < this->bias[0].size(); ++row) {
            this->bias[0][row] += -this->step_size * d_bias[row];
        }

        return d_input;
    }

    // some helper functions.
    size_t layer_size() const {
        return this->size;
    }

    Matrix read_weights() const {
        return this->weight;
    }
};

#endif
