
#ifndef EVOLUTIONAL_DROPOUT_HPP
#define EVOLUTIONAL_DROPOUT_HPP

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <utility>
#include <memory>
#include <thread>
#include <cmath>

#include "layers.hpp"
#include "aux.hpp"
#include "thread_aux.hpp"


/*
 * Have to modify during the test phase. Don't forget...
 */

template<typename Weight>
class EvoDropout2d : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;

    EvoDropout2d(size_t input_size): 
        Layer_2D<Weight>("evolutional dropout"),
        keep_probs(input_size)
    {} 

    EvoDropout2d(EvoDropout2d&& other):
        Layer_2D<Weight>(std::move(other)),
        keep_probs(std::move(other.keep_probs))
    {}

    EvoDropout2d(const EvoDropout2d& other):
        Layer_2D<Weight>(other),
        keep_probs(other.keep_probs)
    {}

    EvoDropout2d* clone() {
        return new EvoDropout2d(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        if(!this->is_training) {
            this->last_output = input;
            return input; // dropout does nothing during test phase!
        }
        Matrix dropped(input.size(), std::vector<Weight>(input[0].size(), 0));
        generate_probs(input);

        std::vector<Weight> mask(input[0].size());

        for(size_t node = 0; node < mask.size(); ++node) {
            mask[node] = static_cast<Weight>(keep(node)) / keep_probs[node];
        }

        for(size_t row = 0; row < input.size(); ++row) {
            for(size_t col = 0; col < input[0].size(); ++col) {
                dropped[row][col] = input[row][col] * mask[col];
            }
        }

        this->last_output = Matrix(1, mask);
        return dropped;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        // same as forward_pass but splits up the indices of the outer
        // for loop when doing element wise product with the mask. 
        this->last_input = input;
        if(!this->is_training) {
            this->last_output = input;
            return input; 
        }
        Matrix dropped(input.size(), std::vector<Weight>(input[0].size(), 0));
        
        generate_probs(input);

        std::vector<Weight> mask(input[0].size());

        for(size_t node = 0; node < mask.size(); ++node) {
            mask[node] = static_cast<Weight>(keep(node)) / keep_probs[node];
        }

        auto rows = thread_alg::split_indices(dropped.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(&EvoDropout2d::async_apply_drop,
                                 std::ref(rows[thread]),
                                 std::ref(mask),
                                 std::ref(input),
                                 std::ref(dropped));
        }
        for(auto& thread : threads) {
            thread.join();
        }

        this->last_output = Matrix(1, mask);

        return dropped;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        // easy! just d_out times the mask, which is stored in last_output 
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size(), 0)); 
        for(size_t i = 0; i < d_out.size(); ++i) {
            for(size_t j = 0; j < d_out[0].size(); ++j) {
                d_input[i][j] = d_out[i][j] * this->last_output[0][j];
            }
        }
        return d_input;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size(), 0)); 
        
        size_t use_threads = (n_threads / 4) + 1; // don't need all the threads
        auto rows = thread_alg::split_indices(d_out.size(), n_threads);
        std::vector<std::thread> threads;
        
        for(size_t thread = 0; thread < use_threads; ++thread) {
            threads.emplace_back(
                [](std::vector<int>& indices,
                   const Matrix& o, // d_out
                   const Matrix& m, // mask (this->last_output
                   Matrix& i) // d_input
                {
                    for(auto& index : indices) {
                        for(size_t col = 0; col < o[0].size(); ++col) {
                            i[index][col] = o[index][col] * m[0][col];
                        }
                    }
                },
                std::ref(rows[thread]),
                std::ref(d_out),
                std::ref(this->last_output),
                std::ref(d_input)
            );
        }

        for(auto& thread : threads) {
            thread.join();
        }
    
        return d_input;
    }

private:
    std::vector<double> keep_probs;

    bool keep(size_t node) {
        return aux::gen_double(0,1) <= keep_probs[node];
    }

    void generate_probs(const Matrix& input) {
        size_t d = keep_probs.size(); // datum size
        size_t m = input.size(); // batch size

        for(size_t i = 0; i < d; ++i) {

            // can't use std::accumulate because its summing comlumn i.
            double numerator = 0;
            for(size_t j = 0; j < m; ++j) {
                numerator += std::pow(input[j][i], 2);
            }
            numerator = std::sqrt(numerator / static_cast<double>(m));
            
            double denominator = 0;
            for(size_t ip = 0; ip < d; ++ip) {
                double piece = 0;
                for(size_t j = 0; j < m; ++j) {
                    piece += std::pow(input[j][ip], 2);
                } 
                denominator += std::sqrt(piece / static_cast<double>(m));
            }
            keep_probs[i] = numerator / denominator;
        }

        // make it so that the keep probs aren't all too tiny:
        // f : [min, max] -> [min_desired, max_desired]
        double max_desired = 0.9;
        double min_desired = 0.4;
        double min_cur = *std::min_element(keep_probs.begin(),keep_probs.end());
        double max_cur = *std::max_element(keep_probs.begin(),keep_probs.end());
        
        for(auto& x : keep_probs) {
            x = (x - min_cur) / 
                (max_cur - min_cur) * (max_desired-min_desired) + min_desired;
        }
    }

    // splits the outer for loops indices.
    static void async_apply_drop(std::vector<int>& rows,
                                 const std::vector<Weight>& mask,
                                 const Matrix& input,
                                 Matrix& to_drop) 
    {
        for(const auto& row : rows) {
            for(size_t col = 0; col < to_drop[0].size(); ++col) {
                to_drop[row][col] = input[row][col] * mask[col];
            }
        }
    }
};

#endif
