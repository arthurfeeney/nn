
#ifndef LRELU_HPP
#define LRELU_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include "../../aux.hpp"
#include "../layers.hpp"


/*
 * Basically the exact same as ReLu (relu.hpp). Only it applies
 *  n > 0 ? n : scale * n.
 */

template <typename Weight = double>
class LRelu : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    LRelu(): Layer_2D<Weight>("lrelu") {}

    LRelu(double scale): Layer_2D<Weight>("lrelu"), scale(scale) {}

    LRelu(LRelu&& other):
        Layer_2D<Weight>(std::move(other)), 
        scale(std::move(other.scale)) 
    {}

    LRelu(const LRelu& other): Layer_2D<Weight>(other), scale(other.scale) {}

    LRelu* clone() {
        return new LRelu(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        this->last_output = leaky_relu(input);
        return this->last_output;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        this->last_input = input;
        this->last_output = async_lrelu(input, n_threads);
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
                // set value in d_out to the scale, 
                // other wise keep it the same.
                // cuz (d/dx scale*x) = scale.
                Weight val = this->last_output[row][col];
                grad[row][col] = val >= 0 ? 1 : scale;
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
        /*
         * just splits up the indices of the outer for loop. Gives each 
         * thread a different set of indices to go over.
         */
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        
        auto rows = thread_alg::split_indices(d_out.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(
            [](std::vector<int>& indices,
                const Matrix& o,
                const Matrix& last_o,
                Matrix& i,
                double s) {
                Matrix grad(i);
                for(auto& index : indices) {
                    for(size_t col = 0; col < o[0].size(); ++col) {
                        Weight val = last_o[index][col];
                        grad[index][col] = val >= 0 ? 1 : s;
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
            std::ref(d_input),
            scale);
        }
        for(auto& thread : threads) {
            thread.join();
        }
        return d_input;   
    }
private:

    double scale = 0.01;

    Matrix leaky_relu(const Matrix& c) { 
        // apply leaky relu to container, return leaky relud container.
        // if n >= 0, return n
        // else return scale * n;
        Matrix leaky_relud(c.size(), std::vector<Weight>(c[0].size()));
        for(size_t i = 0; i < c.size(); ++i) {
            for(size_t j = 0; j < c[0].size(); ++j) {
                leaky_relud[i][j] = c[i][j] >= 0 ? c[i][j] : scale * c[i][j];
            }
        }
        return leaky_relud;
    }

    static void lrelu_thread_task(std::vector<int>& indices, 
                                 const Matrix& A,
                                 Matrix& B,
                                 double s) 
    {
        for(const auto& index : indices) {
            for(size_t i = 0; i < A[0].size(); ++i) {
                B[index][i] = A[index][i] >= 0 ? A[index][i] : s * A[index][i];
            }
        }
    }

    Matrix async_lrelu(const Matrix& A, size_t n_threads) {
        Matrix B(A.size(), std::vector<Weight>(A[0].size()));
        auto rows = thread_alg::split_indices(A.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(&LRelu::lrelu_thread_task,
                                 std::ref(rows[thread]),
                                 std::ref(A),
                                 std::ref(B),
                                 scale);
        }
        for(auto& thread : threads) {
            thread.join();
        }
        return B;
    }
};

#endif
