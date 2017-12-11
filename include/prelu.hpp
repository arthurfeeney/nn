#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include "aux.hpp"
#include "layers.hpp"
#include "thread_aux.hpp"

#ifndef LRELU_HPP
#define LRELU_HPP

template <typename Weight = double>
class PRelu : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    PRelu(): Layer_2D<Weight>("lrelu") {}

    PRelu(double scale, size_t input_size): 
        Layer_2D<Weight>("lrelu"), 
        scales(input_size, scale) 
    {}

    PRelu(PRelu&& other):
        Layer_2D<Weight>(std::move(other)), 
        scale(std::move(other.scale)) 
    {}

    PRelu(const PRelu& other): Layer_2D<Weight>(other), scale(other.scale) {}

    PRelu* clone() {
        return new PRelu(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        this->last_output = prelu(input);
        return this->last_output;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {

    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        
        // compute d_input (should be similiar to lrelu)
        // update scales.

        return d_input;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        
        // same as normal but with parallelization!

        return d_input;   
    }
private:

    std::vector<double> scales;

    Matrix prelu(const Matrix& c) { 
        // apply leaky relu to container, return leaky relud container.
        Matrix prelud(c.size(), std::vector<Weight>(c[0].size()));
        for(size_t i = 0; i < c.size(); ++i) {
            for(size_t j = 0; j < c[0].size(); ++j) {
                prelud[i][j] = c[i][j] >= 0 ? c[i][j] : scales[j] * c[i][j];
            }
        }
        return prelud;
    }

    Matrix async_prelu(const Matrix& A, size_t n_threads) {
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
