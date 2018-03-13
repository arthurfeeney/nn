
/*
* I probably could (should) have used the Eigen library or somthing
* for most of these.
* But, I really wanted to do all of this from scratch, soooo...
* Meh, they're all pretty easy to implement anyway.
*/

#ifndef AUX_HPP
#define AUX_HPP

#include <algorithm>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <iostream>
#include <type_traits>

namespace aux {

    // type function to find rank of a container.
    // doesn't work on things that aren't containers.
    template<class T, size_t N = 1>
    struct type_rank : public 
        std::conditional<
            std::is_fundamental<
                typename std::remove_reference<typename T::value_type>::type
            >::value,
            std::integral_constant<size_t, N>,
            type_rank<typename T::value_type, N + 1>
        >::type
    {};

    template<typename Matrix, typename Scalar>
    Matrix scale_mat(const Matrix& m, Scalar s) {
        Matrix m_copy(m);
        for(auto& row : m_copy) {
            for(auto& val : row) {
                val *= s;
            }
        }
        return m_copy;
    }


    template<typename Matrix>
    Matrix splice(const Matrix& m, size_t start_row, size_t start_col, 
                  size_t size) 
    {
        Matrix out(size, std::vector<double>(size, 0));
        for(size_t row = 0; row < size; ++row) {
            for(size_t col = 0; col < size; ++col) {
                out[row][col] = m[row + start_row][col + start_col];
            }
        }
        return out; 
    }


    // assumes depth of 2...
    template<typename Matrix>
    std::vector<double> flatten_2d(Matrix m)
    {
        
        std::vector<double> flat(0);
        for(const auto& row : m) {
            for(auto item : row) {
                flat.push_back(item);
            }
        }
        return flat;
    }

    template<typename Image>
    std::vector<std::vector<double>> flatten_3d(const Image& image)
    {
        std::vector<std::vector<double>> 
            flat(1);
        
        for(const auto& mat : image) {
            auto line = flatten_2d(mat);
            for(size_t i = 0; i < line.size(); ++i) {
                flat[0].push_back(line[i]);
            }
        }

        return flat;
    }

    // requires a 1xN input.
    template<typename FlatMatrix>
    std::vector<std::vector<std::vector<double>>> 
    unflatten(const FlatMatrix& m, size_t depth, size_t height, size_t width) {
        using Matrix = std::vector<std::vector<double>>;
        using Image = std::vector<Matrix>;


        Image shaped(depth, Matrix(height, std::vector<double>(width, 0)));

        size_t index = 0;
        for(auto& d : shaped) {
            for(auto& h : d) {
                for(auto& w : h) {
                    w = m[0][index];
                    ++index;
                }
            }
        }
          
        return shaped;
    }



    // assumes depth of 2 i.e. vector<vector>, not vector<vector<...>...>
    template<typename Matrix>
    auto relu(const Matrix& c) -> Matrix {
        Matrix relud_c(c.size(), std::vector<double>(c[0].size(), 0));
        for(int i = 0; i < c.size(); ++i) {
            for(int j = 0; j < c[0].size(); ++j) {
                relud_c[i][j] = std::max<double>(c[i][j], 0);
            }
        }
        return relud_c;
    }


    template<typename Vect, typename Out = double> 
    Out dot(const Vect& v1, const Vect& v2) {
        Out d = 0;
        for(size_t i = 0; i < v1.size(); ++i) {
            d += v1[i] * v2[i];
        }
        return d;
    }

    template<typename Matrix>
    auto matmul(const Matrix& m1, const Matrix& m2) -> Matrix {
        if(m1[0].size() != m2.size()) {
            std::cout << "num of cols in m1 != num of rows in m2\n";
            std::cout << m1[0].size() << '\n';
            std::cout << m2.size() << '\n';

        }
        Matrix prod(m1.size(), std::vector<double>(m2[0].size(), 0));
        for(size_t row = 0; row < m1.size(); ++row) {
            for(size_t col = 0; col < m2[0].size(); ++col) {
                for(size_t inner = 0; inner < m2.size(); ++inner) {
                    prod[row][col] += m1[row][inner] * m2[inner][col];
                }
            }
        }
        return prod;
    }

    std::vector<std::vector<double>> empty_matrix(size_t height, size_t width) {
        return std::vector<std::vector<double>>(height, 
                                                std::vector<double>(width, 0));
    }

    template<typename Matrix>
    auto matadd(const Matrix& m1, const Matrix& m2) -> Matrix
    {
        if(m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
            throw("matrix dimension not equal.");
        }
        using sub_type = typename Matrix::value_type;
        Matrix summed(m1.size(), sub_type(m1[0].size(), 0));
        for(size_t row = 0; row < m1.size(); ++row) {
            for(size_t col = 0; col < m1[0].size(); ++col) {
                summed[row][col] = m1[row][col] + m2[row][col];
            }
        }
        return summed;
    }

    template<typename Matrix>
    auto matsub(const Matrix& m1, const Matrix& m2) -> Matrix
    {
        if(m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
            throw("matrix dimension not equal.");
        }
        using sub_type = typename Matrix::value_type;
        Matrix sub(m1.size(), sub_type(m1[0].size(), 0));
        for(size_t row = 0; row < m1.size(); ++row) {
            for(size_t col = 0; col < m1[0].size(); ++col) {
                sub[row][col] = m1[row][col] - m2[row][col];
            }
        }
        return sub;
    }

    template<typename Weight = double>
    auto exp(const std::vector<std::vector<Weight>>& m)
        -> std::vector<std::vector<Weight>> {
        auto out(m);
        for(auto& row : out) {
            for(auto& item : row) {
                item = std::exp(item);
            }
        }
        return out;
    }

    // 2d vector overload of log function.
    template<typename Weight = double>
    auto log(const std::vector<std::vector<Weight>>& m)
        -> std::vector<std::vector<Weight>> {
        auto out(m);
        for(auto& row : out) {
            for(auto& item : row) {
                item = std::log(item);
            }
        }
        return out;
    }

    // 1d vector overload of log function.
    template<typename Weight = double>
    auto log(const std::vector<Weight>& m)
        -> std::vector<Weight> {
        auto out(m);
        for(auto& item : out) {
            item = std::log(item);
        }
        return out;
    }

    template<typename Matrix>
    auto transpose(const Matrix& m) -> Matrix {
        size_t rows = m.size();
        size_t cols = m[0].size();
        Matrix trans(cols, std::vector<double>(rows));
        for(size_t row = 0; row < rows; ++row) {
            for(size_t col = 0; col < cols; ++col) {
                trans[col][row] = m[row][col];
            }
        }
        return trans;
    }

    double gen_double(double lower, double upper) {
        std::random_device(r);
        std::default_random_engine re(r());
        std::uniform_real_distribution<double> unif(lower, upper);
        return unif(re);
    }

}

namespace mat_aux {
    template<typename Weight = double>
    using Matrix = std::vector<std::vector<Weight>>;
    template<typename Weight = double>
    using Image = std::vector<Weight>;

    template<typename Weight = double>
    Matrix<Weight> get_block(const Image<Weight>& input, size_t row_index,
                     size_t col_index, size_t filter_size)
    {
        Matrix<Weight> block(filter_size, std::vector<Weight>(filter_size));
        for(size_t row = row_index; row < filter_size; ++row) {
            for(size_t col = col_index; col < filter_size; ++col) {
                block[row - row_index][col - col_index] = input[row][col];
            }
        }
        return block;
    }

    template<typename Weight = double>
    std::vector<Weight> block_to_col(const Image<Weight>& block) {
        std::vector<Weight> column(block.size() * block[0].size() *
                                   block[0][0].size());
        size_t index = 0;
        for(size_t row = 0; row < block.size(); ++row) {
            for(size_t col = 0; col < block[0].size(); ++col) {
                for(size_t depth = 0; depth < block[0][0].size(); ++depth) {
                    column[index] = block[row][col][depth];
                    ++index;
                }
            }
        }
        return column;
    }

    template<typename Weight = double>
    Matrix<Weight> image_to_col(const Image<Weight>& input, size_t filter_size,
                        size_t stride)
    {
        Matrix<Weight> column_matrix;

        for(size_t row = 0;
            row < input.size() - filter_size;
            row += stride)
        {
            for(size_t col = 0;
                col  < input[0].size() - filter_size;
                col += stride)
            {
                Matrix<Weight> block = get_block(input, row, col, filter_size);
                column_matrix.push_back(block_to_col(block));
            }
        }
        return column_matrix;
    }

    template<typename Weight = double>
    Image<Weight> mat_to_image(const Matrix<Weight>& input, size_t height,
                               size_t width, size_t depth)
    {
        Image<Weight>(height,
                      Matrix<Weight>(width, std::vector<Weight>(depth)));

    }
}

#endif
