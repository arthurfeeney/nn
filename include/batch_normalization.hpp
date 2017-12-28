
#ifndef BATCH_NORMALIZATION_HPP
#define BATCH_NORMALIZATION_HPP

#include <vector>
#include <cmath>

#include "aux.hpp"
#include "layers.hpp"

template<typename Weight = double>
class BatchNorm : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Vect = typename Matrix::value_type;

    BatchNorm(): Layer_2D<Weight>("batchnorm") {}

    BatchNorm(double epsilon, double momentum):
        Layer_2D<Weight>("batchnorm"),
        epsilon(epsilon),
        momentum(momentum),
        // initialize all cached variables
        running_mean(),
        running_var(),
        gamma(),
        beta(),
        mu(),
        xmu(),
        carre(),
        var(),
        sqrtvar(),
        invvar(),
        va2(),
        va3()
    {}

    BatchNorm(const BatchNorm& other):
        Layer_2D<Weight>("batchnorm"),
        epsilon(other.epsilon),
        momentum(other.momentum),
        running_mean(other.running_mean),
        running_var(other.running_var),
        gamma(other.gamma),
        beta(other.beta),
        mu(other.mu),
        xmu(other.xmu),
        carre(other.carre),
        var(other.var),
        sqrtvar(other.sqrtvar),
        invvar(other.invvar),
        va2(other.va2),
        va3(other.va3)
    {}

    BatchNorm* clone() {
        return new BatchNorm(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        size_t batch_size = input.size();
        size_t dimension = input[0].size();

        // running mean and running var not initialized, initialize them!
        if(running_mean.empty()) {
            running_mean = Vect(dimension, 0.0);
        }
        if(running_var.empty()) {
            running_var = Vect(dimension, 0.0);
        }
        if(gamma.empty()) {
            gamma = Vect(dimension, 1.0);
        }
        if(beta.empty()) {
            beta = Vect(dimension, 0.0);
        }

        Matrix out;

        this->last_input = input;

        if(this->is_training) { // train mode
            mu = scale_vect(row_sum(input), 
                            1 / static_cast<double>(batch_size));
            
            xmu = sub_vect_from_rows(input, mu);
            
            carre = pow_matrix(xmu, 2);
            
            var = scale_vect(row_sum(carre),
                             1 / static_cast<double>(batch_size));
            
            sqrtvar = vect_sqrt(add_num_to_vect(var, epsilon));

            invvar = invert_vect(sqrtvar);

            va2 = mult_matrix_row_by_vect(xmu, invvar);

            va3 = mult_matrix_row_by_vect(va2, gamma);

            out = add_vector_to_matrix_rows(va3, beta);

            running_mean = vect_sum(scale_vect(running_mean, momentum),
                           scale_vect(mu, (1.0 - momentum)));

            running_var = vect_sum(scale_vect(running_var, momentum),
                           scale_vect(var, (1.0 - momentum)));
                            
            // all needed values are cached, so forward pass is done.
        }
        else { // test mode.
            mu = running_mean;
            var = running_var;
            Matrix xhat = 
                divide_rows_by_vect(
                        sub_vect_from_rows(input, mu), 
                        vect_sqrt(add_num_to_vect(var, epsilon)));
            out = add_vector_to_matrix_rows(
                    mult_matrix_row_by_vect(xhat, gamma), beta);
        }

        return out;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix async_forward_pass(const Matrix& input, size_t num_threads) {
        return Matrix();
    }

    Matrix backward_pass(const Matrix& input) {
        return input;
    }

    Matrix async_backward_pass(const Matrix& input, size_t num_threads) {
        return Matrix();
    }
private:
    // training phase is bool this->is_training in Layer_2D
    double epsilon;

    double momentum;

    Vect running_mean;

    Vect running_var;

    Vect gamma;

    Vect beta;

    // these values need to be cached for the backward pass.
    Vect mu;
    Matrix xmu;
    Matrix carre;
    Vect var;
    Vect sqrtvar;
    Vect invvar;
    Matrix va2;
    Matrix va3;

    // a shit ton of helper functions that probably need to go somewhere else

    Matrix divide_rows_by_vect(const Matrix& m, const Vect& v) {
        Matrix out(m);
        for(size_t row = 0; row < out.size(); ++row) {
            for(size_t col = 0; col < out[0].size(); ++col) {
                out[row][col] /= v[col];
            }
        }
        return out;
    }

    Vect vect_prod(const Vect& v1, const Vect& v2) {
        Vect out(v1);
        for(size_t i = 0; i < out.size(); ++i) {
            out[i] *= v2[i];
        }
        return out;
    }

    Vect vect_elementwise_divide(const Vect& num, const Vect& denom) {
        Vect out(num);
        for(size_t i = 0; i < out.size(); ++i) {
            out[i] /= denom[i];
        }
        return out;
    }

    Vect vect_sum(const Vect& v1, const Vect& v2) {
        Vect out(v1);
        for(size_t i = 0; i < out.size(); ++i) {
            out[i] += v2[i];
        }
        return out;
    }

    Matrix add_vector_to_matrix_rows(const Matrix& m, const Vect& v) {
        Matrix out(m);
        for(size_t row = 0; row < m.size(); ++row) {
            for(size_t col = 0; col < v.size(); ++row) {
                out[row][col] = m[row][col] + v[col];
            }
        }
        return out;
    }

    Matrix mult_matrix_row_by_vect(const Matrix& m, const Vect& v) {
        Matrix out(m);
        for(size_t row = 0; row < m.size(); ++row) {
            for(size_t col = 0; col < v.size(); ++col) {
                out[row][col] = m[row][col] * v[col];
            }
        }
        return out;
    }

    Vect invert_vect(const Vect& v) {
        Vect out(v);
        for(auto& val : out) {
            val = 1 / val;
        }
        return out;
    }

    Vect vect_sqrt(const Vect& v) {
        Vect out(v);
        for(auto& val : out) {
            val = std::sqrt(val);
        }
        return out;
    }

    Vect add_num_to_vect(const Vect& v, double num) {
        Vect out(v);
        for(auto& val : out) {
            val += num;
        }
        return out;
    }

    Matrix sub_vect_from_rows(const Matrix& m, const Vect& v) {
        Matrix out(m);
        for(size_t row = 0; row < out.size(); ++row) {
            for(size_t col = 0; col < out[0].size(); ++col) {
                out[row][col] -= v[col];
            }
        }
        return out;
    }

    Matrix pow_matrix(const Matrix& m, int power) {
        Matrix out(m);
        for(auto& row : out) {
            for(auto& val : row) {
                val = std::pow(val, power);
            }
        }
        return out;
    }

    Vect row_sum(const Matrix& m) {
        Vect out(m[0].size(), 0.0);
        for(size_t row = 0; row < m.size(); ++row) {
            for(size_t col = 0; col < m[0].size(); ++row) {
                out[col] += m[row][col];
            }
        }
        return out;
    }

    Vect scale_vect(const Vect& v, double scalar) {
        Vect out(v);
        for(auto& val : out) {
            val *= scalar;
        }
        return out;
    }
};

#endif
