
#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

#include "aux.hpp"

#ifndef LAYERS_3D_HPP
#define LAYERS_3D_HPP

// base class for layers that apply 3d transformations.

template<typename Weight = double>
class Layer_3D {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;
    
    Layer_3D(std::string layer_type): layer_type(layer_type) {}

    Layer_3D(std::string layer_type, size_t input_height, size_t input_width,
             size_t input_depth, double learning_rate):
        input_height(input_height), 
        input_width(input_width), 
        input_depth(input_depth),
        layer_type(layer_type),
        step_size(learning_rate)
    {}

    Layer_3D(Layer_3D&& other):
        input_height(other.input_height),
        input_width(other.input_width),
        input_depth(other.input_depth),
        last_input(other.last_input),
        last_output(other.last_output),
        layer_type(other.layer_type),
        step_size(other.step_size)
    {}

    Layer_3D& operator=(Layer_3D&& other) {
        if(this != &other) {
            input_height = other.input_height;
            input_width = other.input_width;
            input_depth = other.input_depth;
            last_input = other.last_input;
            last_output = other.last_output;
            layer_type = other.layer_type; 
            step_size = other.step_size;
        }
        return *this;
    }

    Layer_3D& operator=(const Layer_3D& other) {
        if(this != &other) {
            input_height = other.input_height;
            input_width = other.input_width;
            input_depth = other.input_depth;
            last_input = other.last_input;
            last_output = other.last_output;
            layer_type = other.layer_type; 
            step_size = other.step_size;
        }
        return *this;
    }

    Layer_3D(const Layer_3D& other):
        input_height(other.input_height),
        input_width(other.input_width),
        input_depth(other.input_depth),
        last_input(other.last_input),
        last_output(other.last_output),
        layer_type(other.layer_type),
        step_size(other.step_size)
    {}

    virtual Image forward_pass(const Image& input) = 0;

    virtual Image operator()(const Image& input) = 0;

    virtual Image backward_pass(const Image& d_out) = 0;

    virtual std::tuple<int, int, int> proper_output_dim() const = 0;

    virtual Layer_3D* clone() = 0;

    std::string type() const {
        return layer_type;
    }

    void set_phase(bool training) {
        is_training = training;
    }

protected:
    size_t input_height;
    size_t input_width;
    size_t input_depth;

    Image last_input;
    Image last_output;

    std::string layer_type;

    double step_size;

    bool is_training = true;

};

#endif
