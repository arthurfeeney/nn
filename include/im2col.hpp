

#ifndef IM2COL_HPP
#define IM2COL_HPP

#include <type_traits>
#include <cassert>

#include "aux.hpp"

namespace im2col { 
// these may not work for images that are not square. 


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
    size_t input_width = input[0][0].size();

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

template<typename Image, typename Matr>
static Image 
col_2_im(const Matr& input, size_t kernel_size, size_t stride, 
         size_t output_height, size_t output_width) 
{
    // used by col_2_im_batches for conv2d backward pass
    // input is [M] x [H x W]
    // output is [M] x [H] x [W]

    using ImPlaneType =
        typename std::remove_reference<typename Image::value_type>::type;
    using RowType =
        typename std::remove_reference<typename ImPlaneType::value_type>::type;

    // transposing makes it easier to process the columns.
    Matr input_cols = aux::transpose(input);

    size_t depth = input.size() / (kernel_size*kernel_size);


    Image image(depth,
                ImPlaneType(output_height, 
                            RowType(output_width, 0)));
 
    size_t image_r = 0, image_c = 0;
    for(auto& col : input_cols) {
        
        size_t col_index = 0;
        for(size_t d = 0; d < depth; ++d) {
            for(size_t kh = 0; kh < kernel_size; ++kh) {
                for(size_t kw = 0; kw < kernel_size; ++kw) {
                    image[d][image_r+kh][image_c+kw] = col[col_index];
                }
            }
        }


        image_r += stride; 
        if(image_r > output_height) {
            image_r = 0;
            image_c += stride; 
        }
    } 

    return image;
}


template<typename Images, typename Matr>
Matr im_2_col_batches(const Images& input, size_t kernel_size, size_t stride,
                      size_t output_height, size_t output_width)
{
    
    // assumes images are square.

    // puts a single image in adjacent columns. Doesn't spread them out.
    using Image = 
        typename std::remove_reference<typename Images::value_type>::type;
    using MatrRowType =
        typename std::remove_reference<typename Matr::value_type>::type;

    // images are dxhxw
    size_t num_images = input.size();
    size_t depth = input[0].size();
    
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

        for(size_t row = 0; row < column_size; ++row) {
            size_t im_col = 0;
            for(size_t col = i*one_im_cols; col < (i+1)*one_im_cols; ++col) 
            {
                input_matr[row][col] = im_matr[row][im_col];    
                im_col += 1;
            }
        }


    }

    return input_matr;
}

template<typename ImageBatch, typename Matrix>
ImageBatch
col_2_im_batches(const Matrix& input, size_t kernel_size, size_t stride, 
                 size_t output_height, size_t output_width)
{

    // this assumes that the images are square. 

    using Image = 
        typename std::remove_reference<
            typename ImageBatch::value_type
        >::type;

    ImageBatch output(0);  // output is initally empty.
    size_t num_cols_per_image = output_height * output_width;

    size_t num_images = input[0].size() / num_cols_per_image;

    for(size_t image = 0; image < num_images; ++image) {
        Matrix image_matr = aux::splice(input, 0, image*num_cols_per_image,
                                        num_cols_per_image);        
    /*
        Image im = col_2_im<Image, Matrix>(image_matr, kernel_size, stride, 
                                           output_height, output_width); 
        output.push_back(im);
    */
    }
    return ImageBatch();
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
    // output is [N] x [M] x [H] x [W] batch of images
    // a single image will be in adjecent columns. Not spread out.
    using ImType = 
        typename std::decay<typename ImageBatch::value_type>::type;
    using PlaneType =
        typename std::decay<typename ImType::value_type>::type;
    using RowType = 
        typename std::decay<typename PlaneType::value_type>::type;

    size_t num_images = inputs[0].size() / (height * width);

    ImageBatch image_batch(num_images, 
                           ImType(depth, 
                                   PlaneType(height, 
                                             RowType(width, 0))));

    for(size_t row = 0; row < depth; ++row) {
        size_t col = 0;
        for(size_t im = 0; im < num_images; ++im) {
            for(size_t h = 0; h < height; ++h) {
                for(size_t w = 0; w < width; ++w) {
                    image_batch[im][row][h][w] = inputs[row][col];
                    ++col; 
                }
            }
        }
    }

    return image_batch;
}

template<typename ImageBatch, typename Matrix>
Matrix
image_batch_2_matrix(const ImageBatch& input) 
{
    // input is [N] x [M] x [H] x [W]
    // output is [M] x [N x H x W]
    using RowType = 
        typename std::remove_reference<typename Matrix::value_type>::type;
    size_t num_inputs = input.size();
    size_t num_channels = input[0].size();
    size_t height = input[0][0].size();
    size_t width = input[0][0][0].size();

    //std::cout << num_inputs << ' ' << num_channels << ' ' <<
    //            height << '\n';


    Matrix output(num_channels, RowType(num_inputs * height * width));
    
    
    for(size_t n = 0; n < num_inputs; ++n) {
        size_t output_col = height * width * n;
        for(size_t m = 0; m < num_channels; ++m) {
            for(size_t h = 0; h < height; ++h) {
                for(size_t w = 0; w < width; ++w) {
                    output[m][output_col] = input[n][m][h][w];
                    ++output_col;
                }
            }
        }
    }

    return output;
}

} // namespace im2col
#endif // IM2COL_HPP
