
#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include "../aux.hpp"
#include "layer_3d.hpp"
#include "../im2col.hpp"


/*
 *  Add in bias. filter[i] + bias[i];
 *  It should just be a vector.
 */


template<typename Opt, typename Weight = double>
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
        padding(padding),
        optimizer()
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

        std::tuple<int, int, int> output_dims = proper_output_dim();
        const size_t output_height = std::get<0>(output_dims);
        const size_t output_width = std::get<1>(output_dims);
        const size_t output_depth = std::get<2>(output_dims);
                   
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


        // reshape out_matr into an image.
        ImageBatches images_output = 
            im2col::matrix_2_image_batch<
                ImageBatches,
                Matrix
            >(out_matr, output_height, output_width, output_depth);

        this->last_output = images_output;
        
        return images_output;
    }

    ImageBatches operator()(const ImageBatches& input) {
        return forward_pass(input);
    }

    ImageBatches backward_pass(const ImageBatches& d_out) {

        std::tuple<int, int, int> output_dims = proper_output_dim();
        const size_t output_height = std::get<0>(output_dims);
        const size_t output_width = std::get<1>(output_dims);
        const size_t output_depth = std::get<2>(output_dims);

        // reshape d_out into a matrix.
        Matrix d_out_matr =
            im2col::image_batch_2_matrix<
                ImageBatches, Matrix
            >(d_out);

        
        auto filter_matr = im2col::conv_2_row<Filters, Matrix>(this->filters);

        auto im_matr = 
            im2col::im_2_col_batches<
                ImageBatches, Matrix
            >(this->last_input, filter_size, stride, output_height,
              output_width);
    
         auto d_input_matr = aux::matmul(aux::transpose(filter_matr),
                                         d_out_matr);

         auto d_filters_matr = aux::matmul(d_out_matr, 
                                           aux::transpose(im_matr));
         
        

        optimizer.perform(filter_matr, d_filters_matr, this->step_size);

        this->filters = 
            im2col::matrix_2_image_batch<
                ImageBatches,
                Matrix
            >(filter_matr, output_height, output_width, output_depth);

        
        // d_input = im2col::col_2_im(d_input_matr, kernel_size, stride,
        //                            padding);


         return d_out;
        
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

    Opt optimizer;

};

#endif
