
#ifndef RELU_HPP
#define RELU_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>
#include <thread>
#include <functional>

#include "../../aux.hpp"
#include "../layers.hpp"
#include "../../thread_aux.hpp"


template <typename Weight = double>
class Relu : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Relu(): Layer_2D<Weight>("relu") {}

    Relu(Relu&& other):Layer_2D<Weight>(std::move(other)) {}

    Relu(const Relu& other): Layer_2D<Weight>(other) {}

    Relu* clone() {
        return new Relu(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        this->last_output = relu(input);
        return this->last_output;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        this->last_input = input;
        this->last_output = async_relu(input, n_threads);
        return this->last_output;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        Matrix grad(d_out.size(), std::vector<Weight>(d_out[0].size()));
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                // if value in spot of last_input < 0,
                // set value in d_out to zero, other wise keep it the same.
                // assign value of d_out to d_input.
                Weight val = this->last_output[row][col];
                grad[row][col] = val > 0 ? 1 : 0;
            }
        }
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                d_input[row][col] = grad[row][col] * d_out[row][col];
            }
        }
        return d_input;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        // same as backwad_pass, but it splits up the indices in the 
        // outer for loop so each thread gets its own set of indices
        // to go over.
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        
        auto rows = thread_alg::split_indices(d_out.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(
                [](std::vector<int>& indices,
                    const Matrix& o,
                    const Matrix& last_o,
                    Matrix& i) {
                    Matrix grad(i);
                    for(auto& index : indices) {
                        for(size_t col = 0; col < o[0].size(); ++col) {
                            Weight val = last_o[index][col];
                            grad[index][col] = val > 0 ? 1 : 0;
                        }
                    }
                    for(auto& index : indices) {
                        for(size_t col = 0; col < o[0].size(); ++col) {
                            i[index][col] = grad[index][col] * o[index][col];
                        }
                    }
                },
                std::ref(rows[thread]),
                std::ref(d_out),
                std::ref(this->last_output),
                std::ref(d_input));
        }
        for(auto& thread : threads) {
            thread.join();
        }
        return d_input;   
    }

private:
    Matrix relu(const Matrix& c) { 
        Matrix relud_c(c.size(), std::vector<Weight>(c[0].size()));
        for(size_t i = 0; i < c.size(); ++i) {
            for(size_t j = 0; j < c[0].size(); ++j) {
                relud_c[i][j] = std::max<double>(c[i][j], 0);
            }
        }
        return relud_c;
    }

    
    static void relu_thread_task(std::vector<int>& indices, 
                                 const Matrix& A,
                                 Matrix& B) 
    {
        for(const auto& index : indices) {
            for(size_t i = 0; i < A[0].size(); ++i) {
                B[index][i] = std::max<double>(A[index][i], 0);
            }
        }
    }

    Matrix async_relu(const Matrix& A, size_t n_threads) {
        Matrix B(A.size(), std::vector<Weight>(A[0].size()));
        auto rows = thread_alg::split_indices(A.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(&Relu::relu_thread_task,
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
