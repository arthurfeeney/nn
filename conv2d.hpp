#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

#include "aux.hpp"
#include "layer_3d.hpp"

#ifndef CONV2D_HPP
#define CONV2D_HPP

template<typename Weight = double>
class Conv2d : public Layer_3D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;
    using Filters = std::vector<Image>;

    Conv2d(size_t num_filters, size_t filter_size, size_t stride,
           size_t input_height, size_t input_width, size_t input_depth,
           size_t padding):
            Layer_3D<Weight>("conv2d", input_height, input_width, 
                             input_depth), 
            filter_size(filter_size),
            stride(stride),
            padding(padding),
            num_filters(num_filters),
            filters(num_filters, 
                    Image(filter_size, 
                          Matrix(filter_size, 
                                 std::vector<Weight>(input_depth))))
    {
        for(size_t filter = 0; filter < num_filters; ++filter) {
            for(size_t row = 0; row < filter_size; ++row) {
                for(size_t col = 0; col < filter_size; ++col) {
                    for(size_t depth = 0; depth < this->input_depth; ++depth) {
                        filters[filter][row][col][depth] = 
                            aux::gen_double(-.5, .5);
                    }
                }
            }
        }
    }

    Conv2d(Conv2d&& other):
        Layer_3D<Weight>(other.layer_type, other.input_height, 
                         other.input_width, other.input_depth),
        filters(other.filters),
        filter_size(other.filter_size),
        stride(other.stride),
        padding(other.padding)
    {}

    Conv2d& operator=(Conv2d&& other) {
        if(this != &other) {
            this->layer_type = other.layer_type;
            this->input_height = other.input_type;
            this->input_width = other.input_width;
            this->input_depth = other.input_depth;
            filters = other.filters;
            filter_size = other.filters_size;
            stride = other.stride;
            padding = other.padding;
        }
        return *this;
    }

    Conv2d(const Conv2d& other):
        Layer_3D<Weight>(other.layer_type, other.input_height, 
                         other.input_width, other.input_depth),
        filters(other.filters),
        filter_size(other.filter_size),
        stride(other.stride),
        padding(other.padding) 
    {}

    // highest quality code!
    // filters are height*width*depth. Images are depth*width*height. -____- im dumb.
    Image forward_pass(const Image& input) {
        std::tuple<int, int, int> output_dims = proper_output_dim();
        const size_t output_height = std::get<0>(output_dims);
        const size_t output_width = std::get<1>(output_dims);
        const size_t output_depth = std::get<2>(output_dims);

        Image image_output(output_depth, 
                           Matrix(output_height, 
                                  std::vector<double>(output_width, 0)));

        size_t depth_count = 0;
        for(const auto& filter : filters) {
            for(int depth = 0; depth < this->input_depth; ++depth) {
                // matrix of filter_size slice at depth..
                Matrix filter_at_depth(filter_size, 
                                       std::vector<Weight>(filter_size));
                for(size_t row = 0; row < filter_size; ++row) {
                    for(size_t col = 0; col < filter_size; ++col) {
                        filter_at_depth[row][col] = filter[row][col][depth];
                    }
                }
                // mult filter across image.
                for(size_t row_front = 0, out_row = 0; 
                    row_front < input.size() - filter_size; 
                    row_front += stride, ++out_row) {
                    for(size_t col_front = 0, out_col = 0;
                        col_front < input[0].size() - filter_size; 
                        col_front += stride, ++out_col) 
                    {
                        // get splice of image.
                        Matrix image_splice(filter_size,
                                            std::vector<Weight>(filter_size));
                        for(size_t row = row_front; 
                            row < row_front + filter_size;
                            ++row)
                        {
                            for(size_t col = col_front;
                                col < col_front + filter_size;
                                ++col)
                            {
                                image_splice[row-row_front][col-col_front] =
                                    input[depth][row][col];
                            }
                        }
                        // build outputs from window
                        Matrix window(filter_size, std::vector<Weight>(filter_size));
                        for(int row = 0; row < filter_size; ++row) {
                            for(int col = 0; col < filter_size; ++col) {
                                window[row][col] = image_splice[row][col] * 
                                                    filter_at_depth[row][col];
                            }
                        }
                        Weight elem_window_sum = 0; 
                        for(const auto& row : window) {
                            for(const auto& value : row) {
                                elem_window_sum += value;
                            }
                        }
                        image_output[depth_count][out_row][out_col] += elem_window_sum;
                    }
                }
            }
            ++depth_count;
        }
        return image_output;
    }

    Image operator()(const Image& input) {
        return forward_pass(input);
    }

    Image backward_pass(const Image& d_out) {
        return d_out;
    }
 
    std::tuple<int, int, int> proper_output_dim() const {
        int depth = num_filters;
        int height = (this->input_height - filter_size) / stride;
        int width = (this->input_width - filter_size) / stride;
        return std::tuple<int, int, int>(height, width, depth);
    }

private:
    Filters filters;

    size_t num_filters;
    size_t filter_size;
    size_t stride;

    size_t padding;
};

#endif
