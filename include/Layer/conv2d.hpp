
#ifndef CONV2D_HPP
#define CONV2D_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include <iostream>

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

    Conv2d(size_t num_filters, size_t filter_size, size_t stride,
           size_t input_height, size_t input_width, size_t input_depth,
           size_t padding, double learning_rate):
        
        Layer_3D<Weight>(
            "conv2d", 
            Matrix(num_filters, 
                   std::vector<Weight>(input_depth * 
                                       filter_size *
                                       filter_size)),
            input_height, input_width, 
            input_depth, learning_rate), 
        num_filters(num_filters),
        filter_size(filter_size),
        stride(stride),
        padding(padding),
        optimizer()
    {
        std::cout << "constructing conv2d" << '\n';
    }

    Conv2d(Conv2d&& other):
        Layer_3D<Weight>(other.layer_type, other.filters, other.input_height, 
                         other.input_width, other.input_depth, 
                         other.step_size),
        num_filters(other.num_filters),
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
            num_filters = other.num_filters;
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
        num_filters(other.num_filters),
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
            im2col::im_2_col_batches<ImageBatches, Matrix>(input, 
                                                           filter_size, 
                                                           stride, 
                                                           output_height, 
                                                           output_width);

        
        // Filters are already store in a matrix. No need to reshape them
        // multiply filters by image
        //
        //Computers [M] x [KxKxC] X [KxKxC] x [NxHxW]
        //
        // [M] x [N x H x W] 
        auto out_matr = aux::matmul(this->filters, im_matr);
        
        // reshape out_matr into an image.
        // [N] x [M] x [H] x [W]
        ImageBatches images_output = 
            im2col::matrix_2_image_batch<
                ImageBatches,
                Matrix
            >(out_matr, output_height, output_width, output_depth);

        return images_output;
    }

    ImageBatches operator()(const ImageBatches& input) {
        return forward_pass(input);
    }

    ImageBatches backward_pass(const ImageBatches& d_out) {

        std::tuple<int, int, int> output_dims = proper_output_dim();
        const size_t output_height = std::get<0>(output_dims);
        const size_t output_width = std::get<1>(output_dims);

        // reshape d_out into a matrix.
        Matrix d_out_matr =
            im2col::image_batch_2_matrix<
                ImageBatches, Matrix
            >(d_out);

        
        auto im_matr = 
            im2col::im_2_col_batches<
                ImageBatches, Matrix
            >(this->last_input, filter_size, stride, output_height,
              output_width);
       
         auto d_input_matr = aux::matmul(aux::transpose(this->filters),
                                         d_out_matr);
         
         
         auto d_filters_matr = aux::matmul(d_out_matr, 
                                           aux::transpose(im_matr));
         
        // update layer weights using the optimizer.
        optimizer.perform(this->filters, d_filters_matr, this->step_size);
         
        
        // use col2im to retrieve d_input. 
        /*
        ImageBatches d_input = 
             im2col::col_2_im_batches<ImageBatches, Matrix>(d_input_matr, 
                                                            filter_size, 
                                                            stride,
                                                            output_height,
                                                            output_width);
        */
        return d_out;
    }
 
    std::tuple<int, int, int> proper_output_dim() const {
        size_t p = 2*padding;
        int depth = num_filters;
        int height = (this->input_height - filter_size + p) / stride + 1;
        int width = (this->input_width - filter_size + p) / stride + 1;
        return std::tuple<int, int, int>(height, width, depth);
    }

    Matrix read_filters() const {
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
