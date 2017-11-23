#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include "aux.hpp"
#include "layers.hpp"

#ifndef LRELU_HPP
#define LRELU_HPP

template <typename Weight = double>
class LRelu : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    LRelu(): Layer_2D<Weight>("lrelu") {}

    LRelu(double scale): Layer_2D<Weight>("lrelu"), scale(scale) {}

    LRelu(LRelu&& other):
        Layer_2D<Weight>(std::move(other)), 
        scale(std::move(other.scale)) 
    {}

    LRelu(const LRelu& other): Layer_2D<Weight>(other), scale(other.scale) {}

    LRelu* clone() {
        return new LRelu(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        return leaky_relu(input);
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        // need to implememt.
        return Matrix();
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                // if value in spot of last_input < 0,
                // set value in d_out to the scale, 
                // other wise keep it the same.
                // cuz (d/dx scale*x) = scale.
                Weight val = this->last_input[row][col];
                d_input[row][col] = val >= 0 ? d_out[row][col] : scale;
            }
        }
        return d_input;
    }
private:

    double scale = 0.01;

    Matrix leaky_relu(const Matrix& c) { 
        // apply leaky relu to container, return leaky relud container.
        Matrix leaky_relud(c.size(), std::vector<Weight>(c[0].size()));
        for(size_t i = 0; i < c.size(); ++i) {
            for(size_t j = 0; j < c[0].size(); ++j) {
                leaky_relud[i][j] = c[i][j] >= 0 ? c[i][j] : scale * c[i][j];
            }
        }
        return leaky_relud;
    }
};

#endif
