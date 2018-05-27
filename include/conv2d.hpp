
#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include "aux.hpp"
#include "layer_3d.hpp"
#include "im2col.hpp"


/*
 *  Add in bias. filter[i] + bias[i];
 *  It should just be a vector.
 */


template<typename Weight = double>
class Conv2d : public Layer_3D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;
    using ImageBatches = std::vector<Image>;
    using Filters = std::vector<Image>;

    Conv2d(size_t num_filters, size_t filter_size, size_t stride,
           size_t input_height, size_t input_width, size_t input_depth,
           size_t padding, double learning_rate):
        
        Layer_3D<Weight>(
            "conv2d", 
            Filters(num_filters, 
                    Image(input_depth, 
                          Matrix(filter_size,
                          std::vector<Weight>(filter_size)))),
            input_height, input_width, 
            input_depth, learning_rate),
        
        num_filters(num_filters),
        filter_size(filter_size),
        stride(stride),
        padding(padding)
    {
        //double fan_in = input_height*input_width*input_depth;
        //auto out_dims = proper_output_dim();
        //double fan_out = std::pow(filter_size, 2)*std::get<2>(out_dims);
        double var = 10e-3;//std::sqrt(6.0 / (fan_in + fan_out));
        std::cout << var << '\n';
        
        for(size_t filter = 0; filter < num_filters; ++filter) {
            for(size_t row = 0; row < filter_size; ++row) {
                for(size_t col = 0; col < filter_size; ++col) {
                    for(size_t depth = 0; depth < this->input_depth; ++depth) {
                        double val = aux::gen_double(-var, var); 
                        std::cout << val << '\n';
                        this->filters[filter][depth][row][col] = val;
                    }
                }
            }
        }
    }

    Conv2d(Conv2d&& other):
        Layer_3D<Weight>(other.layer_type, other.filters, other.input_height, 
                         other.input_width, other.input_depth, 
                         other.step_size),
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
            this->filters = other.filters;
            filter_size = other.filters_size;
            stride = other.stride;
            padding = other.padding;
        }
        return *this;
    }

    Conv2d(const Conv2d& other):
        Layer_3D<Weight>(other.layer_type, other.filters, other.input_height, 
                         other.input_width, other.input_depth, 
                         other.step_size),
        filter_size(other.filter_size),
        stride(other.stride),
        padding(other.padding) 
    {}

    Conv2d* clone() {
        return new Conv2d(*this);
    }

    ImageBatches forward_pass(const ImageBatches& input) {
        this->last_input = input;

        const size_t num_inputs = input.size();

        std::tuple<int, int, int> output_dims = proper_output_dim();
        const size_t output_height = std::get<0>(output_dims);
        const size_t output_width = std::get<1>(output_dims);
        const size_t output_depth = std::get<2>(output_dims);

        ImageBatches images_output(num_inputs, 
            Image(output_depth, 
                  Matrix(output_height, 
                         std::vector<Weight>(output_width, 0)))
        );
        
        
        
        // make image patches into columns of matrix
        // [K x K x C] x [H x W]
        auto im_matr = 
            im2col::im_2_col_batches<
                ImageBatches, Matrix
            >(input, filter_size, stride, output_height, output_width);
        
        // flatten filters into rows of matrix.
        // [M] x [K x K x C]
        auto filter_matr = im2col::conv_2_row<Filters, Matrix>(this->filters);

        // multiply filters by image
        // [M] x [H x W]
        auto out_matr = aux::matmul(filter_matr, im_matr);

        for(size_t i = 0; i < num_inputs; ++i) {
            for(size_t d = 0; d < output_depth; ++d) {
                for(size_t h = 0; h < output_height; ++h) {
                    for(size_t w = 0; w < output_width; ++w) {
                         
                    }
                }
            }
        }


        // reshape out_matr into an image.
        

        this->last_output = images_output;
        

        return images_output;
    }

    ImageBatches operator()(const ImageBatches& input) {
        return forward_pass(input);
    }

    // BREAK THIS UP INTO MORE FUNCTIONS. 
    ImageBatches backward_pass(const ImageBatches& d_out) {

        return d_out;
        num_filters = this->filters.size();
        
        // will need to pad this->last_input when I add in padding.
        
    }
 
    std::tuple<int, int, int> proper_output_dim() const {
        size_t p = 2*padding;
        int depth = this->filters.size();
        int height = (this->input_height - filter_size + p) / stride + 1;
        int width = (this->input_width - filter_size + p) / stride + 1;
        return std::tuple<int, int, int>(height, width, depth);
    }

    Filters read_filters() const {
        return this->filters;
    }


private:

    size_t num_filters;
    size_t filter_size;
    size_t stride;

    size_t padding;

};

#endif
