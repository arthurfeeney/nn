
#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <string>
#include <cmath>

#include "../aux.hpp"
#include "../loss.hpp"
#include "../Initializer/uniform.hpp"

template <typename Weight = double>
class Layer_2D {
protected:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Matrix weights;
    Matrix bias;
    size_t size;
    size_t input_size;

    Matrix last_input;
    Matrix last_output;

    std::string layer_type;

    bool is_training = true;

    double step_size = 1e-3;

    bool weighted = false;

public:
    Layer_2D(std::string layer_type): layer_type(layer_type) {}

    Layer_2D(std::string layer_type, double learning_rate):
        layer_type(layer_type),
        step_size(learning_rate)
    {}

    // layer sets sizes of stuff and the name of the layer.
    Layer_2D(size_t num_nodes, size_t input_size, std::string layer_type,
             double learning_rate):
        // set weights to random values. KxN
        weights(input_size, std::vector<Weight>(num_nodes, 0)),
        // bias initially zero. 1xN
        bias(1, std::vector<Weight>(num_nodes, 0)),
        size(num_nodes),
        input_size(input_size),
        layer_type(layer_type),
        step_size(learning_rate),
        weighted(true)
    {}

    ~Layer_2D() = default;

    Layer_2D(Layer_2D&& other):
        weights(other.weights),
        bias(other.bias),
        size(other.size),
        input_size(other.input_size),
        last_input(other.last_input),
        last_output(other.last_output), 
        is_training(other.is_training),
        weighted(other.weighted)
    {
        other.size = 0;
        other.input_size = 0;
    }

    Layer_2D& operator=(Layer_2D&& other) {
        if(this != &other) {
            weights = other.weights;
            bias = other.bias;
            size = other.size;
            last_input = other.last_input;
            last_output = other.last_output;
            input_size = other.input_size;
            is_training = other.is_training;
            weighted = other.weighted;
        }
        return *this;
    }

    Layer_2D(const Layer_2D& other):
        weights(other.weights),
        bias(other.bias),
        size(other.size),
        input_size(other.input_size),
        last_input(other.last_input),
        last_output(other.last_output), 
        is_training(other.is_training),
        weighted(other.weighted)
    {}

    virtual Matrix forward_pass(const Matrix& input) = 0;

    virtual Matrix async_forward_pass(const Matrix& input, size_t nt) = 0;

    virtual Matrix backward_pass(const Matrix& input) = 0;

    virtual Matrix async_backward_pass(const Matrix& input, size_t nt) = 0;

    virtual Matrix operator()(const Matrix& input) = 0;

    virtual Layer_2D* clone() = 0;

    void change_phase() {
        is_training = !is_training;
    }

    bool current_phase() {
        return is_training;
    }

    void set_phase(bool training) {
        is_training = training;
    }

    Matrix get_weights() const {
        return weights;
    }

    std::string type() const {
        return layer_type;
    }

    Layer_2D<Weight>& operator+=(const Layer_2D<Weight>& other) {
        if(!weighted) return *this;
        for(size_t row = 0; row < other.weights.size(); ++row) {
            for(size_t col = 0; col < other.weights[0].size(); ++col) {
                weights[row][col] += other.weights[row][col];
            }
        }
        for(size_t row = 0; row < other.bias[0].size(); ++row) {
            bias[0][row] += other.bias[0][row];
        }
        return *this;
    }

    Layer_2D<Weight>& operator/=(const size_t count) {
        if(!weighted) return *this;
        for(size_t row = 0; row < weights.size(); ++row) {
            for(size_t col = 0; col < weights[0].size(); ++col) {
                weights[row][col] /= count;
            }
        }
        for(size_t row = 0; row < bias[0].size(); ++row) {
            bias[0][row] /= count;
        }
        return *this;     
    }

    void initialize(std::string which_init, double gain = 1) {
        // this function is not ideal at all, but it was convenient :/
        // would rather have init::<function> passed in instead of a string
        size_t fan_in = weights.size();
        size_t fan_out = weights[0].size();

        if(which_init == "xavier_uniform") {
            double denom = static_cast<double>(fan_in + fan_out);
            double a = gain * std::sqrt(6.0 / denom); 
            init::rand::UniformRand<Weight> init(-a, a);
            for(size_t i = 0; i < weights.size(); ++i) {
                for(size_t j = 0; j < weights[i].size(); ++j) {
                    weights[i][j] = init();
                }
            }
        }

        else if(which_init == "xavier_normal") {
            double denom = static_cast<double>(fan_in + fan_out);
            double std = gain * std::sqrt(2.0 / denom);
            init::rand::NormalRand<Weight> init(0, std);
            for(size_t i = 0; i < weights.size(); ++i) {
                for(size_t j = 0; j < weights[i].size(); ++j) {
                    weights[i][j] = init();
                }
            }
        }
    }
};

template<typename Weight = double>
struct Loss_Cross_Entropy {

    using Matrix = std::vector<std::vector<Weight>>;

    Loss_Cross_Entropy() {}

    Weight comp_loss(const Matrix& scores, const Matrix& actual) {
        return loss(scores, actual).second;
    }

    // functor, no need for constructor!
    Matrix forward_pass(const Matrix& scores, const Matrix& actual) {
        auto probs_and_loss = loss(scores, actual);
        most_recent_loss = probs_and_loss.second;
        return probs_and_loss.first;
    }

    Matrix operator()(const Matrix& scores, const Matrix& actual) {
        return forward_pass(scores, actual);
    }

    Matrix backward_pass(const Matrix& probs, const Matrix& actual) {
        return dloss(probs, actual);
    }

    Matrix comp_d_loss(const Matrix& scores, const Matrix& actual) {
        Matrix probs = forward_pass(scores, actual);
        Matrix d_loss = backward_pass(probs, actual);
        return d_loss;
    }

    double get_loss() const {
        return most_recent_loss;
    }
private:
    double most_recent_loss = 0;
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

template<typename Weight = double>
class MaxPool2d : public Layer_2D<Weight> {
public:

    using Matrix = std::vector<std::vector<Weight>>;

    using Image = std::vector<Matrix>;

    MaxPool2d(std::initializer_list<Weight> dimensions) {

    }

    Image forward_pass(const Image& input) {

    }

    Image backward_pass(const Image& d_out) {

    }
private:

    Image last_input;
    Image last_output;
};



#endif
