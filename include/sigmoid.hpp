
#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <cmath>

#include "aux.hpp"
#include "thread_aux.hpp"
#include "layers.hpp"

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
        Matrix grad = d_sigify(this->last_input);
        Matrix d_input(grad.size(), std::vector<Weight>(grad[0].size()));
        for(size_t i = 0; i < d_input.size(); ++i) {
            for(size_t j = 0; j < d_input[0].size(); ++j) {
                d_input[i][j] = grad[i][j] * d_out[i][j];
            }
        }
        return d_input;
    }

    Matrix async_backward_pass(const Matrix& input, size_t n_threads) {
        size_t use_threads = (n_threads / 4) + 1;
        if(use_threads == 1) {
            return backward_pass(input);
        }
        this->last_input = input;
        this->last_output = async_helper(input, use_threads,
                                         backward_thread_task);
        return this->last_output;
    }

private:
    Matrix sigify(const Matrix& input) {
        // applies sigmoid function to all elements of input.
        Matrix apply_tanh(input);
        for(auto& row : apply_tanh) {
            for(auto& val : row) {
                val = 1 / (1 + std::exp(val));
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
                B[index][i] = 1 / (1 + std::exp(val));
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
