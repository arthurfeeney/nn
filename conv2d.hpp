#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

#include "aux.hpp"
#include "layers.hpp"

#ifndef CONV2D_HPP
#define CONV2D_HPP

template<typename Weight = double>
class Conv2d : public Layer<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;

    /*
     * add a bool for padding eventually.
     */
    Conv2d(size_t num_filters, size_t filter_size, size_t stride,
           size_t height, size_t width, size_t depth):
            Layer<Weight>(num_filters,
                          filter_size * filter_size * input_depth,
                          "conv2d"),
            filter_size(filter_size),
            stride(stride),
            row_length(filter_size * filter_size * input_depth),
            input_height(height), input_width(width), input_depth(depth)
    {
        for(int row = 0; row < this->size; ++row) {
            for(int col = 0; col < row_length; ++col) {
                this->weights[row][col] = aux::gen_double(-.1, .1);
            }
        }
    }


    Matrix forward_pass(const Matrix& col_matrix) {
        this->last_input = col_matrix;
        this->last_output = aux::matmul(this->weights, col_matrix);
        return this->last_output;
    }

    Matrix operator()(const Matrix& col_matrix) {
        return forward_pass(col_matrix);
    }

    Matrix backward_pass(const Matrix& d_out) {

    }

    // used to reshape output.
    std::tuple<int, int, int> proper_output_dim() const {
        int height = (input_height - filter_size) / (stride + 1);
        int width = (input_width - filter_size) / (stride + 1);
        int depth = this->size;
        return std::tuple<int, int, int>(height, width, depth);
    }

private:
    // this->size is the number of filters.
    size_t filter_size;
    size_t stride;

    size_t input_height;
    size_t input_width;
    size_t input_depth;

    size_t row_length;
};

#endif
