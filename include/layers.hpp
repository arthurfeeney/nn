
#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

#include "aux.hpp"
#include "loss.hpp"

#ifndef LAYERS_HPP
#define LAYERS_HPP

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

public:
    Layer_2D(std::string layer_type): layer_type(layer_type) {}

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
        step_size(learning_rate)
    {}

    ~Layer_2D() = default;

    Layer_2D(Layer_2D&& other):
        weights(other.weights),
        bias(other.bias),
        size(other.size),
        input_size(other.input_size),
        last_input(other.last_input),
        last_output(other.last_output), 
        is_training(other.is_training)
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
        is_training(other.is_training)
    {}

    virtual Matrix forward_pass(const Matrix& input) = 0;

    virtual Matrix backward_pass(const Matrix& input) = 0;

    virtual Matrix operator()(const Matrix& input) = 0;

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
