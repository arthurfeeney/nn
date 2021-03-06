
#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <cmath>

#include "../../aux.hpp"
#include "../../thread_aux.hpp"
#include "../layers.hpp"

template<typename Weight = double>
class Sigmoid : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Sigmoid(): Layer_2D<Weight>("sigmoid") {}

    Sigmoid(Sigmoid&& other): Layer_2D<Weight>(std::move(other)) {}

    Sigmoid(const Sigmoid& other): Layer_2D<Weight>(other) {}

    Sigmoid* clone() {
        return new Sigmoid(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        this->last_output = sigify(input);
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
        Matrix grad = d_sigify(this->last_output);
        Matrix d_input(grad.size(), std::vector<Weight>(grad[0].size()));
        for(size_t i = 0; i < d_input.size(); ++i) {
            for(size_t j = 0; j < d_input[0].size(); ++j) {
                d_input[i][j] = grad[i][j] * d_out[i][j];
            }
        }
        return d_input;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        size_t use_threads = n_threads; //(n_threads / 4) + 1;
        if(use_threads == 1) {
            return backward_pass(d_out);
        }
        Matrix grad = async_helper(this->last_output, use_threads, 
                                   backward_thread_task);

        Matrix d_input(grad.size(), std::vector<Weight>(grad[0].size()));

        auto rows = thread_alg::split_indices(grad.size(), use_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < use_threads; ++thread) {
            threads.emplace_back(&Sigmoid::comp_d_input,
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
    Matrix sigify(const Matrix& input) {
        // applies sigmoid function to all elements of input.
        Matrix apply_tanh(input);
        for(auto& row : apply_tanh) {
            for(auto& val : row) {
                val = 1 / (1 + std::exp(-val));
            }
        }
        return apply_tanh;
    }

    Matrix d_sigify(const Matrix& input) {
        // gradient of sigify function.
        Matrix sub_matrix(input);
        for(auto& row : sub_matrix) {
            for(auto& val : row) {
                val = val*(1 - val);
            }
        }
        return sub_matrix;
    }

    static void forward_thread_task(std::vector<int>& indices,
                                    const Matrix& A,
                                    Matrix& B)
    {
        for(const auto& index : indices) {
            for(size_t i = 0; i < A[0].size(); ++i) {
                Weight val = A[index][i];
                B[index][i] = 1 / (1 + std::exp(-val));
            }
        }
    }

    static void backward_thread_task(std::vector<int>& indices,
                                     const Matrix& A,
                                     Matrix& B)
    {
        for(const auto& index : indices) {
            for(size_t i = 0; i < A[0].size(); ++i) {
                Weight val = A[index][i];
                B[index][i] = val*(1 - val);
            }
        }
    }

    static void comp_d_input(std::vector<int>& indices,
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
        // This function was actually a really good idea...
        // Should have done this sooner... w/e tho.
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
