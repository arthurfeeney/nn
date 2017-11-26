
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
    using Filters = std::vector<Image>;

    Layer_3D(std::string layer_type): layer_type(layer_type) {}

    Layer_3D(std::string layer_type, Filters filters, size_t input_height, 
             size_t input_width, size_t input_depth, double learning_rate):
        filters(filters),
        input_height(input_height), 
        input_width(input_width), 
        input_depth(input_depth),
        layer_type(layer_type),
        step_size(learning_rate)
    {}

    Layer_3D(Layer_3D&& other):
        filters(other.filters),
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
            filters = other.filters;
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
            filters = other.filters;
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
        filters(other.filters),
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

    virtual Layer_3D<Weight>* clone() = 0;

    Layer_3D<Weight>& operator+=(const Layer_3D<Weight>& other) {
        if(this == &other) {
            return *this;
        }
        for(size_t filter = 0; filter < filters.size(); ++filter) {
            for(size_t height = 0; height < filters[0].size(); ++height) {
                for(size_t width = 0; width < filters[0][0].size(); ++width) {
                    for(size_t depth = 0; 
                        depth < filters[0][0][0].size(); 
                        ++depth) 
                    {
                        filters[filter][height][width][depth] +=
                            other.filters[filter][height][width][depth];
                    }
                }
            }    
        }
        return *this;
    }

    Layer_3D<Weight>& operator/=(const size_t scalar) {
        for(size_t filter = 0; filter < filters.size(); ++filter) {
            for(size_t height = 0; height < filters[0].size(); ++height) {
                for(size_t width = 0; width < filters[0][0].size(); ++width) {
                    for(size_t depth = 0; 
                        depth < filters[0][0][0].size(); 
                        ++depth) 
                    {
                        filters[filter][height][width][depth] /= scalar;
                    }
                }
            }    
        }
        return *this;
    }

    std::string type() const {
        return layer_type;
    }

    void set_phase(bool training) {
        is_training = training;
    }

protected:
    Filters filters;

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
