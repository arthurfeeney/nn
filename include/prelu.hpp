
#ifndef PRELU_HPP
#define PRELU_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>
#include <cmath>

#include "aux.hpp"
#include "layers.hpp"
#include "thread_aux.hpp"


template <typename Opt, typename Weight = double>
class PRelu : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    PRelu(): Layer_2D<Weight>("prelu") {}

    PRelu(size_t input_size, double learning_rate): 
        Layer_2D<Weight>("prelu", learning_rate),
        optimizer(),
        scales(input_size) 
    {
        double v = std::sqrt(6.0 / input_size);
        for(auto& val : scales) {
            val = aux::gen_double(0, v);
        }
    }

    PRelu(PRelu&& other):
        Layer_2D<Weight>(std::move(other)), 
        scales(std::move(other.scales)) 
    {}

    PRelu(const PRelu& other): Layer_2D<Weight>(other), scales(other.scales) {}

    PRelu* clone() {
        return new PRelu(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        this->last_output = prelu(input);
        return this->last_output;
    }

    Matrix async_forward_pass(const Matrix& input, size_t n_threads) {
        this->last_input = input;
        this->last_output = async_prelu(input, n_threads);
        return this->last_output;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                Weight val = this->last_input[row][col];
                d_input[row][col] = val >= 0 ? 
                                        d_out[row][col] : 
                                        d_out[row][col] * scales[col];
            }
        }   

        std::vector<double> feature_sums(scales.size(), 0.0);
        for(size_t i = 0; i < this->last_output[0].size(); ++i) {
            for(size_t e = 0; e < this->last_output.size(); ++e) {
                double activation = this->last_output[e][i];
                double gradient_of_activation = activation > 0 ? activation : 0;
                feature_sums[i] += d_out[e][i] * gradient_of_activation; 
            }
        }

        // wrapped in another vector because optimizer.perform requires
        // a matrix type. 
        std::vector<std::vector<double>> scales_matr(1, scales);
        std::vector<std::vector<double>> feature_sums_matr(1, feature_sums);

        optimizer.perform(scales_matr, feature_sums_matr, this->step_size);

        for(size_t i = 0; i < scales.size(); ++i) {
            scales[i] = scales_matr[0][i];
        }

        return d_input;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t n_threads) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));

        auto rows = thread_alg::split_indices(d_out.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(
                [](std::vector<int>& indices,
                   std::vector<double> scales,
                   const Matrix& o,
                   const Matrix& last_i,
                   Matrix& i) 
                {
                    for(auto& index : indices) {
                        for(size_t col = 0; col < o[0].size(); ++col) {
                            Weight val = last_i[index][col];
                            i[index][col] = val >= 0 ? 
                                                o[index][col] :
                                                o[index][col] * scales[col];
                        }
                    }
                },
                std::ref(rows[thread]),
                scales,
                std::ref(d_out),
                std::ref(this->last_input),
                std::ref(d_input));
        }
        for(auto& thread : threads) {
            thread.join();
        }
        std::vector<double> feature_sums(scales.size(), 0.0);

        auto rows_again = thread_alg::split_indices(this->last_output[0].size(),
                                         n_threads);

        std::vector<std::thread> threads_again;
        
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads_again.emplace_back(
                [](std::vector<int>& indices,
                   std::vector<double> scales,
                   const Matrix& o,
                   const Matrix& last_o,
                   std::vector<double>& feature) 
                {
                    for(auto& i : indices) {
                        for(size_t e = 0; e < last_o.size(); ++e) {
                            double activation = last_o[e][i];
                            double gradient_of_activation = 
                                activation > 0 ? activation : 0;
                            feature[i] += o[e][i] * gradient_of_activation;
                        }
                    }
                },
                std::ref(rows_again[thread]),
                scales,
                std::ref(d_out),
                std::ref(this->last_output),
                std::ref(feature_sums));
        }

        for(auto& thread : threads_again) {
            thread.join();
        }

        std::vector<std::vector<double>> scales_matr(1, scales);
        std::vector<std::vector<double>> feature_sums_matr(1, feature_sums);

        optimizer.perform(scales_matr, feature_sums_matr, this->step_size);

        for(size_t i = 0; i < scales.size(); ++i) {
            scales[i] = scales_matr[0][i];
        }

        return d_input;   
    }

private:
    Opt optimizer;

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

    static void prelu_thread_task(std::vector<int>& indices,
                                  std::vector<double> scales,
                                  const Matrix& A,
                                  Matrix& B)
    {
        for(const auto& index : indices) {
            for(size_t i = 0; i < A[0].size(); ++i) {
                Weight val = A[index][i];
                B[index][i] = val > 0 ? val : val * scales[i];
            }
        }
    }

    Matrix async_prelu(const Matrix& A, size_t n_threads) {
        Matrix B(A.size(), std::vector<Weight>(A[0].size()));
        auto rows = thread_alg::split_indices(A.size(), n_threads);
        std::vector<std::thread> threads;
        for(size_t thread = 0; thread < n_threads; ++thread) {
            threads.emplace_back(&PRelu::prelu_thread_task,
                                 std::ref(rows[thread]),
                                 scales,
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
