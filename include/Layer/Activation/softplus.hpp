
#ifndef SOFTPLUS_HPP
#define SOFTPLUS_HPP

#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <cmath>

#include "../../aux.hpp"
#include "../../thread_aux.hpp"
#include "../layers.hpp"

template<typename Weight = double>
class Softplus : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Softplus(): Layer_2D<Weight>("tanh") {}

    Softplus(Softplus&& other): Layer_2D<Weight>(std::move(other)) {}

    Softplus(const Softplus& other): Layer_2D<Weight>(other) {}

    Softplus* clone() {
        return new Softplus(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        this->last_output = softplus(input);
        return this->last_output;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        // it will be fast enough with one thread size this is so small.
        size_t use_threads = (n_threads / 4) + 1;
        if(use_threads == 1) {
            return forward_pass(input);
        }
        this->last_input = input;
        this->last_output = async_helper(input, use_threads,
                                         forward_thread_task);
        return this->last_output;
    }

    Matrix backward_pass(const Matrix& d_out) {
        /*
         * tanh'(z) = 1 - tanh^2(z).
         * so you can just do matrix_of_ones - matrix_from_forward_pass !
         */
        Matrix grad = d_softplus(this->last_output);
        Matrix d_input(grad.size(), std::vector<Weight>(grad[0].size()));
        for(size_t i = 0; i < d_input.size(); ++i) {
            for(size_t j = 0; j < d_input[0].size(); ++j) {
                d_input[i][j] = grad[i][j] * d_out[i][j];
            }
        }
        return d_input;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        size_t use_threads = (n_threads / 4) + 1;
        if(use_threads == 1) {
            return backward_pass(d_out); // if only one thread, just use normal
        }
        Matrix grad = async_helper(this->last_output, use_threads,
                                   backward_thread_task);
        
        Matrix d_input(grad.size(), std::vector<Weight>(grad[0].size()));

        auto rows = thread_alg::split_indices(grad.size(), use_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < use_threads; ++thread) {
            threads.emplace_back(&Softplus::comp_d_input,
                                 std::ref(rows[thread]),
                                 std::ref(grad),
                                 std::ref(d_out),
                                 std::ref(d_input));
        }

        for(auto& thread : threads) {
            thread.join();
        }
        
        return d_input;
    }

private:
    Matrix softplus(const Matrix& input) {
        Matrix apply_tanh(input);
        for(auto& row : apply_tanh) {
            for(auto& val : row) {
                val = std::log1p(std::exp(val));
            }
        }
        return apply_tanh;
    }

    Matrix d_softplus(const Matrix& input) {
        Matrix sub_matrix(input);
        for(auto& row : sub_matrix) {
            for(auto& val : row) {
                val = 1 / (1 + std::exp(-val));
            }
        }
        return sub_matrix;
    }

    static void 
    forward_thread_task(std::vector<int>& indices,
                        const Matrix& A,
                        Matrix& B)
    {
        for(const auto& index : indices) {
            for(size_t i = 0; i < A[0].size(); ++i) {
                B[index][i] = std::log1p(std::exp(A[index][i]));
            }
        }
    }

    static void 
    backward_thread_task(std::vector<int>& indices,
                         const Matrix& A,
                         Matrix& B)
    {
        for(const auto& index : indices) {
            for(size_t i = 0; i < A[0].size(); ++i) {
                B[index][i] = 1 / (1 + std::exp(-A[index][i]));
            }
        }
    }

    static void 
    comp_d_input(std::vector<int>& indices,
                 const Matrix& grad,
                 const Matrix& d_out,
                 Matrix& d_input)
    {
        for(const auto& row : indices) {
            for(size_t col = 0; col < d_input[0].size(); ++col) {
                d_input[row][col] = grad[row][col] * d_out[row][col];
            }
        }
    }

    
    template<typename Func>
    Matrix async_helper(const Matrix& A, size_t n_threads, Func f) {
        Matrix B(A);
        // this task is so simple I should definitely not use all the threads.
        auto rows = thread_alg::split_indices(A.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(f,
                                 std::ref(rows[thread]),
                                 std::ref(A),
                                 std::ref(B));
        }
        for(auto& thread : threads) {
            thread.join();
        }
        return B;
    }

};

#endif
