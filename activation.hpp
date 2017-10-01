#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

#include "aux.hpp"
#include "layers.hpp"

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

template <typename Weight = double>
struct Relu : public Layer_2D<Weight> {
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Relu(): Layer_2D<Weight>("relu") {}

    Relu(Relu&& other):Layer_2D<Weight>(std::move(other)) {}

    Relu(const Relu& other): Layer_2D<Weight>(other) {}

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        return aux::relu(input);
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                // if value in spot of last_input < 0,
                // set value in d_out to zero, other wise keep it the same.
                // assign value of d_out to d_input.
                Weight val = this->last_input[row][col];
                d_input[row][col] = val >= 0 ? d_out[row][col] : 0;
            }
        }
        return d_input;
    }
};

#endif
