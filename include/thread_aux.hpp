#include <vector>
#include <thread>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <functional>

#ifndef THREAD_AUX_HPP
#define THREAD_AUX_HPP

namespace thread_alg {

    template<typename Matrix>
    static void 
    comp_matr_sub(const std::vector<int>& rows,
                  const std::vector<int>& cols,
                  const Matrix& A,
                  const Matrix& B,
                  Matrix& C) 
    {
        for(const auto& row : rows) {
            for(const auto& col : cols) {
                for(size_t inner = 0; inner < B.size(); ++inner) {
                    C[row][col] += A[row][inner] * B[inner][col];
                }
            }
        }
    } 

    std::vector<std::vector<int>>
    split_indices(size_t size, size_t num_splits) {
        std::vector<std::vector<int>> indices(num_splits, std::vector<int>(0));
        for(size_t i = 0; i < size; ++i) {
            indices[i % num_splits].push_back(i);
        }
        return indices; 
    }

    template<typename Matrix>
    Matrix matmul(const Matrix& A, const Matrix& B, size_t n_threads) {

        if(A[0].size() != B.size()) {
            std::cout << "async_matmul: rows != cols \n";
            std::cout << A[0].size() << '\n';
            std::cout << B.size() << '\n';
        }
        
        using value_type = typename Matrix::value_type;
        Matrix C(A.size(), value_type(B[0].size()));
        
        size_t row_split_count = n_threads - (n_threads / 2);
        size_t col_split_count = n_threads / 2;

        auto rows = split_indices(A.size(), row_split_count);
        auto cols = split_indices(B[0].size(), col_split_count);
        
        std::vector<std::thread> threads;
        
        for(size_t row = 0; row < row_split_count; ++row) {
            for(size_t col = 0; col < col_split_count; ++col) {
                threads.emplace_back(
                        &comp_matr_sub<Matrix>,
                        std::ref(rows[row]),
                        std::ref(cols[col]),
                        std::ref(A),
                        std::ref(B),
                        std::ref(C));
            }
        }

        for(auto& thread : threads) {
            thread.join();
        }
        return C;
    }    
 
}

#endif
