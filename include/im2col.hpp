

#ifndef IM2COL_HPP
#define IM2COL_HPP

#include <type_traits>
#include <cassert>

#include "aux.hpp"

namespace im2col {



template<typename Im, typename Matr>
static Matr 
im_2_col(const Im& input, size_t kernel_size, size_t stride, 
         size_t output_height, size_t output_width)
{
    // takes a single image
    // input must be depth * height * width.

    using ImPlaneType = 
        typename std::remove_reference<typename Im::value_type>::type;   // 2d  
       
    using MatrRowType = 
        typename std::remove_reference<typename Matr::value_type>::type; // 1d

    size_t depth = input.size();
    size_t input_height = input[0].size();
    size_t input_width = input[1].size();

    size_t column_size = kernel_size * kernel_size * depth;
    size_t num_cols = output_height*output_width;

    Matr input_matr(column_size,
                    MatrRowType(num_cols, 0));
    
    size_t c = 0;
    for(size_t h = 0; h < input_height-kernel_size+1; h += stride) {
        for(size_t w = 0; w < input_width-kernel_size+1; w += stride) {
            size_t r = 0; // which row insertion begins at.
            for(size_t d = 0; d < depth; ++d) {
                auto region = aux::splice(input[d], h, w, kernel_size); 
                auto col = aux::flatten_2d<ImPlaneType, MatrRowType>(region);
                
                for(size_t i = 0; i < kernel_size*kernel_size; ++i) {
                    input_matr[r+i][c] = col[i];
                }
                r += kernel_size*kernel_size; // move r below inserted 
            }
            c += 1; // scoot over insertion to the next column
        }
    }

    return input_matr; 
}

template<typename Im, typename Matr>
static Im 
col_2_im(const Matr& input, size_t kernel_size, size_t stride) {
    // used by col_2_im_batcehs for conv2d backward pass
    // input is [M] x [H x W]
    // returns a single image.
    
}


template<typename Images, typename Matr>
Matr im_2_col_batches(const Images& input, size_t kernel_size, size_t stride,
                      size_t output_height, size_t output_width)
{
    // puts a single image in adjacent columns. Doesn't spread them out.
    using Image = 
        typename std::remove_reference<typename Images::value_type>::type;
    using MatrRowType =
        typename std::remove_reference<typename Matr::value_type>::type;

    // images are dxhxw
    size_t num_images = input.size();
    size_t depth = input[0][0].size();
    
    size_t column_size = kernel_size * kernel_size * depth;
    size_t num_cols = num_images * output_height * output_width;

    size_t one_im_cols = output_height * output_width;

    Matr input_matr(column_size, MatrRowType(num_cols, 0));
    
    for(size_t i = 0; i < num_images; ++i) {
        auto im_matr = 
            im2col::im_2_col<
                Image,
                Matr
            >(input[i], kernel_size, stride, output_height, output_width);
        for(size_t r = 0; r < column_size; ++r) {
            for(size_t c = i*one_im_cols, sub_c = 0; 
                c < (i+1)*one_im_cols;
                ++c, ++sub_c) 
            {
                input_matr[r][c] = im_matr[r][sub_c];
            }
        }
    }
    return input_matr;
}


template<typename Conv, typename Matr>
Matr conv_2_row(const Conv& c) {
    // c is a container of filters
    // returns [M]x[k x k x c] matrix
    // THIS IS NOT kn2row. This just flattens kernels into rows of a matrix.
    
    using RowType = 
        typename std::remove_reference<typename Matr::value_type>::type; // 1d

    size_t num_rows = c.size();
    size_t num_cols = c[0].size() * c[0][0].size() * c[0][0].size();
    Matr row_matr(num_rows, RowType(num_cols, 0));
 
    for(size_t k = 0; k < num_rows; ++k) {
        auto flat_kernel = aux::flatten_3d(c[k]);
        for(size_t column = 0; column < num_cols; ++column) {
            row_matr[k][column] = flat_kernel[0][column];
        }
    }

    return row_matr;
}

template<typename ImageBatch, typename Matrix>
ImageBatch 
matrix_2_image_batch(const Matrix& inputs, size_t height, size_t width, 
                     size_t depth) {
    // inputs is [M] x [N x H x W]
    // outpust a [N] x [M] x [H] x [W] batch of images
    // a single image will be in adjecent columns. Not spread out.
    using ImType = 
        typename std::remove_reference<typename ImageBatch::value_type>::type;
    using PlaneType =
        typename std::remove_reference<typename ImType::value_type>::type;
    using RowType = 
        typename std::remove_reference<typename PlaneType::value_type>::type;

    size_t one_images_columns = height * width;

    size_t num_images = inputs[0].size() / (one_images_columns);


    ImageBatch image_batch(num_images, 
                           ImType(depth, 
                                   PlaneType(height, 
                                             RowType(width, 0))));

    for(size_t i = 0; i < num_images; ++i) {
        for(size_t d = 0; d < inputs.size(); ++d) {
            size_t c = 0;
            for(size_t h = 0; h < height; ++h) {
                for(size_t w = 0; w < width; ++w) {
                    image_batch[i][d][h][w] = inputs[d][i*c];
                    ++c;
                }
            }
        }
    }
    return image_batch;

}

} // namespace im2col
#endif // IM2COL_HPP
