 
#include <algorithm>
#include <vector>
#include <utility>

#include "aux.hpp"
#include "loss.hpp"

#ifndef LAYERS_HPP
#define LAYERS_HPP


template <typename Weight = double>
class Layer {
protected:
    using Matrix = std::vector<std::vector<Weight>>;
    
    Matrix weights;
    Matrix bias;
    int size;    
    int input_size;

    Matrix last_input;
    Matrix last_output;

public:
    Layer() {}
    
    Layer(int num_nodes, int input_size):
        // set weights to random values. KxN
        weights(input_size, std::vector<Weight>(num_nodes, 0)), 
        // bias initially zero. 1xN
        bias(1, std::vector<Weight>(num_nodes, 0)),
        size(num_nodes), 
        input_size(input_size) 
    {}
    
    ~Layer() = default;
    
    Layer(Layer&& other):
        weights(other.weights),
        bias(other.bias),
        size(other.size),
        input_size(other.input_size),
        last_input(other.last_input),
        last_output(other.last_output) {
        other.size = 0;
        other.input_size = 0;      
    }

    Layer& operator=(Layer&& other) {
        if(this != &other) {
            weights = other.weights;
            bias = other.bias;
            size = other.size;
            last_input = other.last_input;
            last_output = other.last_output;
            input_size = other.input_size;  
        }
        return *this;
    }

    Layer(const Layer& other): 
        weights(other.weights),
        bias(other.bias),
        size(other.size),
        input_size(other.input_size),
        last_input(other.last_input),
        last_output(other.last_output) {}
    
    Layer& operator=(const Layer& other) {
            
    }

    virtual Matrix forward_pass(const Matrix& input) = 0;
    
    virtual Matrix backward_pass(const Matrix& input) = 0;
    
    virtual Matrix operator()(const Matrix& input) = 0;

    Matrix get_weights() {
        return weights;
    }

};

template<typename Weight = double>
struct Loss_Cross_Entropy {
    using Matrix = std::vector<std::vector<Weight>>;
    
    Loss_Cross_Entropy() {}

    double comp_loss(const Matrix& scores, const Matrix& actual) {
        return loss(scores, actual).second;
    }

    // functor, no need for constructor!
    Matrix forward_pass(const Matrix& scores, const Matrix& actual) {
        return loss(scores, actual).first;
    }

    Matrix operator()(const Matrix& scores, const Matrix& actual) {
        return forward_pass(scores, actual);
    }

    Matrix backward_pass(const Matrix& probs, const Matrix& actual) {
        return dloss(probs, actual);
    }
        
    Matrix comp_d_loss(const Matrix& scores, const Matrix& actual) {
        Matrix&& probs = forward_pass(scores, actual);
        Matrix&& d_loss = backward_pass(probs, actual);
        return d_loss;
    }

};

// very incomplete. Can't do regression rn. :(
template<typename Weight = double>
struct Squared_Error_Loss {
    using Matrix = std::vector<std::vector<Weight>>;
    
    Squared_Error_Loss() {}

    double comp_loss(double pred, double actual) {
        return (pred - actual) * (pred - actual);
    }
    
    double forward_pass(double pred, double actual) {
        return comp_loss(pred, actual);
    }
    
    double operator()(double pred, double actual) {
        return comp_loss(pred, actual);
    }

    double backward_pass(double loss, double actual) {
        return 1.0;
    }
};

template <typename Weight = double>
class Relu : public Layer<Weight> {
private:
    using Matrix = std::vector<std::vector<Weight>>;

public:

    Relu(): Layer<Weight>() {}

    Relu(Relu&& other):Layer<Weight>(std::move(other)) {}

    Relu(const Relu& other): Layer<Weight>(other) {}

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        return aux::relu(input);
    }
    
    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        Matrix d_input(d_out.size(), std::vector<Weight>(d_out[0].size()));
        for(size_t row = 0; row < d_out.size(); ++row) {
            for(size_t col = 0; col < d_out[0].size(); ++col) {
                // if value in spot of last_input < 0, 
                // set value in d_out to zero, other wise keep it the same.
                // assign value of d_out to d_input. 
                Weight val = this->last_input[row][col];
                d_input[row][col] = val >= 0 ? d_out[row][col] : 0;
            }
        }
        return d_input;
    }
};

// treated as a functor.
template <typename Weight = double>
class Dense : public Layer<Weight> {
private:
    using Matrix = std::vector<std::vector<Weight>>;
public:
    Dense():Layer<Weight>() {}
    // constructor sets size and default values of weights.  
    Dense(int num_nodes, int input_size):
        Layer<Weight>(num_nodes, input_size)
    {
        for(auto& row : this->weights) {
            for(auto& item : row) {
                item = aux::gen_double(-0.1, 0.1);
            }
        }
    }
    
    ~Dense() = default; // nothing to delete, really
    
    Dense(Dense&& other): Layer<Weight>(std::move(other)) {}
    
    Dense(const Dense& other): Layer<Weight>(other) {}

    Matrix forward_pass(const Matrix& input) {
        // input is 1xK (a one by I matrix).
        this->last_input = input;
        // produces a 1xN matrix. From 1xK, KxN
        Matrix apply_weights = aux::matmul(input, this->weights); 
        // adds bias to 1xN matrix;
        Matrix apply_bias = aux::matadd(apply_weights, this->bias);
        // saves the output of layer.
        this->last_output = apply_bias;
        return apply_bias; //returns the scores
    }

    Matrix operator()(const Matrix& input) {
        return forward_pass(input);
    }

    Matrix backward_pass(const Matrix& d_out) {
        /*
        * updates weights and bias. Returns gradient of input.
        * requires:
        * - last_input: data input to network. 1xI
        * - dout: derivative for backprop.
        * - weights: layers weights.KxN
        * - bias: layers bias. 1xN
        * finds gradients for weights, bias, and input to feed to next
        *  layer.
        */        
        double step_size = 1e-3;
        auto d_weights = aux::matmul(aux::transpose(this->last_input), 
                                     d_out);
        
        std::vector<double> d_bias(d_out.size());
        for(size_t row = 0; row < d_out.size(); ++row) {
            d_bias[row] = std::accumulate(d_out[row].begin(), 
                                          d_out[row].end(), 0.0);
        }

        auto d_input = aux::matmul(d_out, aux::transpose(this->weights));
    
        for(size_t row = 0; row < this->weights.size(); ++row) {
            for(size_t col = 0; col < this->weights[0].size(); ++col) {
                this->weights[row][col] += -step_size * d_weights[row][col];
            }
        } 

        // bias is a matrix, d_bias is a vector (just cuz im dumb.)
        
        for(size_t row = 0; row < this->bias.size(); ++row) {
            this->bias[row][0] += -step_size * d_bias[row];
        }

        return d_input; // used for next layer. 
    }
};

#endif
