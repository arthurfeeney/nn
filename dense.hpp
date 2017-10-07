
#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

#include "aux.hpp"
#include "layers.hpp"

#ifndef DENSE_HPP
#define DENSE_HPP


/*
 * make it so that the range of initial values can (optionally?) be passed in
 */

template <typename Weight = double>
struct Dense : public Layer_2D<Weight> {
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Dense():Layer_2D<Weight>("dense") {}
    // constructor sets size and default values of weights.
    Dense(int num_nodes, int input_size):
        Layer_2D<Weight>(num_nodes, input_size, "dense")
    {
        for(auto& row : this->weights) {
            for(auto& item : row) {
                item = aux::gen_double(-0.1, 0.1);
            }
        }
    }

    ~Dense() = default;

    Dense(Dense&& other): Layer_2D<Weight>(std::move(other)) {}

    Dense(const Dense& other): Layer_2D<Weight>(other) {}

    Matrix forward_pass(const Matrix& input) {
        // computes output of a layer based on input and its weights.
        // input is 1xK (a one by I matrix).
        this->last_input = input;
        // produces a 1xN matrix. From 1xK, KxN
        Matrix apply_weights = aux::matmul(input, this->weights);
        // adds bias to 1xN matrix;
        Matrix apply_bias = aux::matadd(apply_weights, this->bias);
        // saves the output of layer.
        this->last_output = apply_bias;
        return apply_bias; //returns the scores
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
        double step_size = 1e-3; // learning rate
        auto d_weights = aux::matmul(aux::transpose(this->last_input),
                                     d_out);

        std::vector<double> d_bias(d_out.size());
        for(size_t row = 0; row < d_out.size(); ++row) {
            d_bias[row] = std::accumulate(d_out[row].begin(),
                                          d_out[row].end(), 0.0);
        }

        auto d_input = aux::matmul(d_out, aux::transpose(this->weights));

        for(size_t row = 0; row < this->weights.size(); ++row) {
            for(size_t col = 0; col < this->weights[0].size(); ++col) {
                this->weights[row][col] += -step_size * d_weights[row][col];
            }
        }

        // bias is a matrix, d_bias is a vector (just cuz im dumb.)
        for(size_t row = 0; row < this->bias.size(); ++row) {
            this->bias[row][0] += -step_size * d_bias[row];
        }

        return d_input; // used for next layer.
    }

    size_t layer_size() const {
        return this->size;
    }

    Matrix read_weights() const {
        return this->weight;
    }
};

#endif
