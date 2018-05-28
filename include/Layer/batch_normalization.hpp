
#ifndef BATCH_NORMALIZATION_HPP
#define BATCH_NORMALIZATION_HPP

#include <vector>
#include <cmath>
#include <iostream>

#include "../aux.hpp"
#include "layers.hpp"

/*
 * This is not tested very well. It executes and doesn't hurt accuracy too much
 * (worse accuracy doesn't necessarily mean it doesn't work) on small networks
 * Not distributed at all, and afaik positive effects aren't very noticeable
 * on small networks. So a single threaded, taxing op on a big network probably
 * running on my laptop since it's single threaded it wouldn't make sense to 
 * run it on hpc. 
 */


template<typename Opt, typename Weight = double>
class BatchNorm : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Vect = typename Matrix::value_type;

    BatchNorm(): Layer_2D<Weight>("batchnorm") {}

    BatchNorm(double epsilon, double momentum):
        Layer_2D<Weight>("batchnorm"),
        gamma_optimizer(),
        beta_optimizer(),
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
        gamma_optimizer(),
        beta_optimizer(),
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

    BatchNorm(BatchNorm&& other):
        Layer_2D<Weight>("batchnorm"),
        gamma_optimizer(),
        beta_optimizer(),
        epsilon(std::move(other.epsilon)),
        momentum(std::move(other.momentum)),
        running_mean(std::move(other.running_mean)),
        running_var(std::move(other.running_var)),
        gamma(std::move(other.gamma)),
        beta(std::move(other.beta)),
        mu(std::move(other.mu)),
        xmu(std::move(other.xmu)),
        carre(std::move(other.carre)),
        var(std::move(other.var)),
        sqrtvar(std::move(other.sqrtvar)),
        invvar(std::move(other.invvar)),
        va2(std::move(other.va2)),
        va3(std::move(other.va3))
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

        Matrix out(batch_size, std::vector<Weight>(dimension, 0.0));

        this->last_input = input;

        if(this->is_training) { // train mode
            // step 1
            mu = scale_vect(row_sum(input), 
                            1 / static_cast<double>(batch_size));
            
            // step 2
            xmu = sub_vect_from_rows(input, mu);
            
            // step 3
            carre = pow_matrix(xmu, 2);
            
            // step 4
            var = scale_vect(row_sum(carre),
                             1 / static_cast<double>(batch_size));
            
            // step 5
            sqrtvar = vect_sqrt(add_num_to_vect(var, epsilon));

            // step 6
            invvar = invert_vect(sqrtvar);

            // step 7
            va2 = mult_matrix_row_by_vect(xmu, invvar);

            // step 8
            va3 = mult_matrix_row_by_vect(va2, gamma);

            // step 9
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
            // (input - mu) / sqrt(var + eps)
            Matrix xhat = 
                divide_rows_by_vect(
                        sub_vect_from_rows(input, mu), 
                        vect_sqrt(add_num_to_vect(var, epsilon)));
            out = add_vector_to_matrix_rows(
                    mult_matrix_row_by_vect(xhat, gamma), beta);
        }
        
        this->last_output = out;
        return this->last_output;
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix async_forward_pass(const Matrix& input, size_t num_threads) {
        return Matrix();
    }

    Matrix backward_pass(const Matrix& d_out) {
        size_t batch_size = d_out.size();

        // backprop steps:
        // step 9
        Matrix dva3 = d_out;
        Vect dbeta = row_sum(d_out);

        // step 8
        Matrix dva2 = mult_matrix_row_by_vect(dva3, gamma); 

        Vect dgamma = row_sum(matr_elem_prod(dva3, va2)); //row_sum(dva3 * va2)

        // step 7
        Matrix dxmu = mult_matrix_row_by_vect(dva2, invvar);
        Matrix tmp2(dva2.size(), std::vector<Weight>(dva2[0].size()));
        for(size_t i = 0; i < tmp2.size(); ++i) {
            for(size_t j = 0; j < tmp2[0].size(); ++j) {
                tmp2[i][j] = xmu[i][j] * dva2[i][j];
            }
        } 
        Vect dinvvar = row_sum(tmp2);

        // step 6
        Vect dsqrtvar = 
            scale_vect(invert_vect(vect_prod(pow_vect(sqrtvar, 2), dinvvar)),
                       -1); 

        // step 5
        Vect dvar = 
            vect_prod(scale_vect(pow_vect(add_num_to_vect(var, epsilon), -0.5),
                                 0.5),
                      dsqrtvar);
        
        // step 4
        Matrix tmp3(carre.size(), std::vector<Weight>(carre[0].size(), 1.0));

        Matrix dcarre = 
            mult_matrix_row_by_vect(
                aux::scale_mat(tmp3, 1.0/static_cast<double>(batch_size)),
                dvar);

        //step 3
        //dxmu += 2 * xmu + dcarre
        dxmu = aux::matadd(dxmu, matr_elem_prod(aux::scale_mat(xmu, 2),
                                                dcarre));
        
        //step 2
        Matrix dx = dxmu;
        Vect dmu = scale_vect(row_sum(dxmu), -1.0);

        // step 1
        // dx += (1 / batchsize) * Ones * dmu
        Matrix ones(dxmu.size(), std::vector<Weight>(dxmu[0].size(), 1.0)); 
        dx = aux::matadd(dx,
                         aux::scale_mat(mult_matrix_row_by_vect(ones, dmu), 
                                        1 / static_cast<double>(batch_size)));

        Matrix gamma_matr(1, gamma);
        Matrix dgamma_matr(1, dgamma);
        gamma_optimizer.perform(gamma_matr, dgamma_matr, this->step_size);
        
        Matrix beta_matr(1, beta);
        Matrix dbeta_matr(1, dbeta);
        beta_optimizer.perform(beta_matr, dbeta_matr, this->step_size);

        for(size_t i = 0; i < gamma.size(); ++i) {
            gamma[i] = gamma_matr[0][i];
            beta[i] = beta_matr[0][i];
        }

        return dx;
    }

    Matrix async_backward_pass(const Matrix& d_out, size_t num_threads) {
        return Matrix();
    }

private:
    // two optimizers since I think they need to be separate, since 
    // some optimzers store things which would depend on input
    Opt gamma_optimizer;
    Opt beta_optimizer;

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
    
    // elementwise product of matrices
    Matrix matr_elem_prod(const Matrix& m1, const Matrix& m2) {
        Matrix out(m1); 
        for(size_t row = 0; row < out.size(); ++row) {
            for(size_t col = 0; col < out[0].size(); ++col) {
                out[row][col] *= m2[row][col];
            }
        }
        return out;
    }

    Matrix divide_rows_by_vect(const Matrix& m, const Vect& v) {
        Matrix out(m);
        for(size_t row = 0; row < out.size(); ++row) {
            for(size_t col = 0; col < out[0].size(); ++col) {
                out[row][col] /= v[col];
            }
        }
        return out;
    }

    // elementwise product of vectors
    Vect vect_prod(const Vect& v1, const Vect& v2) {
        Vect out(v1);
        for(size_t i = 0; i < out.size(); ++i) {
            out[i] *= v2[i];
        }
        return out;
    }

    // elementwise division of vectors
    Vect vect_elementwise_divide(const Vect& num, const Vect& denom) {
        Vect out(num);
        for(size_t i = 0; i < out.size(); ++i) {
            out[i] /= denom[i];
        }
        return out;
    }

    // elementwise sum of vectors
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
            for(size_t col = 0; col < v.size(); ++col) {
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

    // vect = 1 / vect
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

    Vect pow_vect(const Vect& m, int power) {
        Vect out(m);
        for(auto& val : out) {
            val = std::pow(val, power);
        }
        return out;
    }

    Vect row_sum(const Matrix& m) {
        Vect out(m[0].size(), 0.0);
        for(size_t row = 0; row < m.size(); ++row) {
            for(size_t col = 0; col < m[0].size(); ++col) {
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
