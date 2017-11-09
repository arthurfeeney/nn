#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include "aux.hpp"
#include "layer_3d.hpp"

#ifndef CONV2D_HPP
#define CONV2D_HPP

/*
 *  Add in bias. filter[i] + bias[i];
 *  It should just be a vector.
 */



template<typename Weight = double>
class Conv2d : public Layer_3D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;
    using Filters = std::vector<Image>;

    Conv2d(size_t num_filters, size_t filter_size, size_t stride,
           size_t input_height, size_t input_width, size_t input_depth,
           size_t padding, double learning_rate):
            Layer_3D<Weight>("conv2d", input_height, input_width, 
                             input_depth, learning_rate), 
            filters(num_filters, 
                    Image(filter_size, 
                          Matrix(filter_size, 
                                 std::vector<Weight>(input_depth)))),
            num_filters(num_filters),
            filter_size(filter_size),
            stride(stride),
            padding(padding)
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
                         other.input_width, other.input_depth, other.step_size),
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
                         other.input_width, other.input_depth, other.step_size),
        filters(other.filters),
        filter_size(other.filter_size),
        stride(other.stride),
        padding(other.padding) 
    {}

    Conv2d* clone() {
        return new Conv2d(*this);
    }

    // highest quality code!
    // filters are height*width*depth. Images are depth*height*width. -____- im dumb.
    Image forward_pass(const Image& input) {
        this->last_input = input;
 
        if(padding > 0) {
            // pad last_input;
            Image tmp(this->input_depth, 
                      Matrix(this->input_height + 2*padding, 
                             std::vector<Weight>(this->input_width + 2*padding, 0)));
            for(size_t d = 0; d < this->input_depth; ++d) {
                for(size_t h = padding; h < this->input_height; ++h) {
                    for(size_t w = padding; w < this->input_width; ++w) {
                        tmp[d][h][w] = this->last_input[d][h-padding][w-padding];
                    }
                }
            }
            this->last_input = tmp;
        } 

        std::tuple<int, int, int> output_dims = proper_output_dim();
        const size_t output_height = std::get<0>(output_dims);
        const size_t output_width = std::get<1>(output_dims);
        const size_t output_depth = std::get<2>(output_dims);

        Image image_output(output_depth, 
                           Matrix(output_height, 
                                  std::vector<Weight>(output_width, 0)));
        
        for(size_t out_depth = 0; out_depth < filters.size(); ++out_depth) {
            auto& filter = filters[out_depth];
            for(size_t depth = 0; depth < this->input_depth; ++depth) {
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
                    row_front < this->last_input[0].size() - filter_size + 1; 
                    row_front += stride, ++out_row) {
                    for(size_t col_front = 0, out_col = 0;
                        col_front < this->last_input[0][0].size() - filter_size + 1; 
                        col_front += stride, ++out_col) 
                    {
                         
                        // get window/splice of input image.
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
                                    this->last_input[depth][row][col];
                            }
                        }
                        
                        // build outputs from window. Elem-wise prod of window and filt
                        Matrix window(filter_size, std::vector<Weight>(filter_size));
                        for(size_t row = 0; row < filter_size; ++row) {
                            for(size_t col = 0; col < filter_size; ++col) {
                                window[row][col] = image_splice[row][col] * 
                                                    filter_at_depth[row][col];
                            }
                        }
                        
                        // sum of values computed in window.
                        Weight elem_window_sum = 0; 
                        for(const auto& row : window) {
                            for(const auto& value : row) {
                                elem_window_sum += value;
                            }
                        }
                        
                        image_output[out_depth][out_row][out_col] = elem_window_sum;
                        
                    }
                }
            } 
        }
        this->last_output = image_output;
        return image_output;
    }

    Image operator()(const Image& input) {
        return forward_pass(input);
    }

    // BREAK THIS UP INTO MORE FUNCTIONS. 
    Image backward_pass(const Image& d_out) {
        // need to account for padding.
        /*
         * need to compute:
         * dx - gradient with respect to input.
         * dw - gradient with respect to weights.
         */
        
        // will need to pad this->last_input when I add in padding.
        
        std::tuple<int, int, int> output_dims = proper_output_dim();
        const size_t output_height = std::get<0>(output_dims);
        const size_t output_width = std::get<1>(output_dims);

        Filters d_filters(num_filters, 
                          Image(filter_size, 
                                Matrix(filter_size,
                                       std::vector<Weight>(this->input_depth, 0))));
        // well written ^^^^ 
       
        // compute gradients of filters.
    
        for(size_t fprime = 0; fprime < num_filters; ++fprime) {
            for(size_t cprime = 0; cprime < this->input_depth; ++cprime) {
                for(size_t i = 0; i < filter_size; ++i) {
                    for(size_t j = 0; j < filter_size; ++j) {
                        
                        // generate piece of the input.
                        Matrix sub_input(output_height,
                                         std::vector<Weight>(output_width, 0)); 
                        
                        // literally no idea if this is right...
                        for(size_t height = i; 
                            height < output_height; 
                            ++height) 
                        {
                            for(size_t width = j; 
                                width < output_width; 
                                ++width) 
                            {
                                sub_input[height][width] = 
                                    this->last_input[cprime]
                                                    [height*stride]
                                                    [width*stride];    
                            }
                        }
                        
                        // fill d_filters with element-wise sum of product of regions.
                        int window_sum = 0;
                        for(size_t row = 0; row < output_height; ++row) {
                            for(size_t col = 0; col < output_width; ++col) {
                                window_sum += d_out[fprime][row][col] * sub_input[row][col];
                            }
                        }
                        d_filters[fprime][i][j][cprime] = window_sum;
                    }
                }
            }
        }

        // update filters.
        for(size_t filter = 0; filter < num_filters; ++filter) {
            for(size_t height = 0; height < filter_size; ++height) {
                for(size_t width = 0; width < filter_size; ++width) {
                    for(size_t depth = 0; depth < this->input_depth; ++depth) {
                        filters[filter][height][width][depth] += 
                            this->step_size *
                            d_filters[filter][height][width][depth];
                    }
                }
            }
        }

        Image d_input(this->input_depth, 
                      Matrix(this->input_height,
                             std::vector<Weight>(this->input_width, 0)));

        // im dumb and made images h*w*d here and not d*h*w like normal. 
        // I also should probably break this up into way more functions.
        for(size_t i = 0; i < this->input_height; ++i) {
            for(size_t j = 0; j < this->input_width; ++j) {
                for(size_t f = 0; f < num_filters; ++f) {
                    for(size_t k = 0; k < output_height; ++k) {
                        for(size_t l = 0; l < output_width; ++l) {
                            Image mask1(filter_size, 
                                        Matrix(filter_size,
                                               std::vector<Weight>(this->input_depth, 0)));
                            Image mask2(filter_size, 
                                        Matrix(filter_size,
                                               std::vector<Weight>(this->input_depth, 0)));
                            
                            if((i + padding - k * stride) < filter_size &&
                               (i + padding - k * stride) >= 0) 
                            {
                                for(size_t width = 0; width < filter_size; ++width) {
                                    for(size_t depth = 0; depth < this->input_depth; ++depth) {
                                        mask1[i + padding - k * stride][width][depth] = 1.0;
                                    }
                                }
                            }
                            if((j + padding - l * stride) < filter_size &&
                               (j + padding - l * stride) >= 0)
                            {
                                for(size_t height = 0; height < filter_size; ++height) {
                                    for(size_t depth = 0; depth < this->input_depth; ++depth) {
                                        mask2[height][j + padding - l * stride][depth] = 1.0;
                                    }
                                }
                            }
                            
                            Image tmp_prod(filter_size,
                                           Matrix(filter_size,
                                                  std::vector<Weight>(this->input_depth, 
                                                                      0)));

                            for(size_t row = 0; row < filter_size; ++row) {
                                for(size_t col = 0; col < filter_size; ++col) {
                                    for(size_t depth = 0; depth < this->input_depth; ++depth) {
                                        tmp_prod[row][col][depth] = 
                                            filters[f][row][col][depth] *
                                            mask1[row][col][depth] * 
                                            mask2[row][col][depth];
                                    }
                                }
                            }
                            std::vector<Weight> w_masked(this->input_depth, 0);
                            for(size_t row = 0; row < filter_size; ++row) {
                                for(size_t col = 0; col < filter_size; ++col) {
                                    for(size_t depth = 0; depth < this->input_depth; ++depth) {
                                        w_masked[depth] += tmp_prod[row][col][depth];
                                    }
                                }
                            }
                            for(size_t depth = 0; depth < this->input_depth; ++depth) {
                                d_input[depth][i][j] += w_masked[depth] * d_out[f][k][l];
                            }
                        }
                    }
                }
            }
        }
        // return image needs to be d*h*w for consistency.
        return d_input;
    }
 
    std::tuple<int, int, int> proper_output_dim() const {
        int depth = filters.size();
        int height = (this->input_height - filter_size + (2 * padding)) / stride + 1;
        int width = (this->input_width - filter_size + (2 * padding)) / stride + 1;
        return std::tuple<int, int, int>(height, width, depth);
    }

    Filters read_filters() const {
        return this->filters;
    }

private:
    Filters filters;

    size_t num_filters;
    size_t filter_size;
    size_t stride;

    size_t padding;

    Image pad_image(const Image& image) {
        // function to pad image (same as thing in forward_pass).
    }
};

#endif
