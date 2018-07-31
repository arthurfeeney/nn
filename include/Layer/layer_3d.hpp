
#ifndef LAYERS_3D_HPP
#define LAYERS_3D_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>

#include "../aux.hpp"
#include "../Initializer/uniform.hpp"

// base class for layers that apply 3d transformations.

template<typename Weight = double>
class Layer_3D {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;
    using ImageBatches = std::vector<Image>;

    Layer_3D(std::string layer_type): layer_type(layer_type) {}

    Layer_3D(std::string layer_type, Matrix filters, size_t input_height, 
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

    Layer_3D<Weight>& operator+=(const Layer_3D<Weight>& other) {
        if(this == &other) {
            return *this;
        }
        for(size_t filter = 0; filter < filters.size(); ++filter) {
            for(size_t idx = 0; idx < filters[0].size(); ++idx) {
                filters[filter][idx] += other.filters[filter][idx];
            }    
        }
        return *this;
    }

    Layer_3D<Weight>& operator/=(const size_t scalar) {
        for(size_t filter = 0; filter < filters.size(); ++filter) {
            for(size_t idx = 0; idx < filters[0].size(); ++idx) {
                filters[filter][idx] /= scalar;
            }    
        }
        return *this;
    }

    void initialize(std::string which_init, double gain = 1) {
        std::pair<size_t, size_t> fans = fan_in_fan_out();
        size_t fan_in = fans.first;
        size_t fan_out = fans.second;

        double denom = static_cast<double>(fan_in + fan_out);

        if(which_init == "xavier_uniform") {

            double std = gain * std::sqrt(2.0 / denom);
            double a = std::sqrt(3.0) * std;
            init::rand::UniformRand<Weight> init(-a, a);
                    
            for(auto& filter : filters) {
                for(auto& val : filter) {
                    val = init();
                    std::cout << val << '\n';
                }
            } 
        }
    }

    std::string type() const {
        return layer_type;
    }

    void set_phase(bool training) {
        is_training = training;
    }

    virtual ImageBatches forward_pass(const ImageBatches& input) = 0;

    virtual ImageBatches operator()(const ImageBatches& input) = 0;

    virtual ImageBatches backward_pass(const ImageBatches& d_out) = 0;

    virtual std::tuple<int, int, int> proper_output_dim() const = 0;

    virtual Layer_3D<Weight>* clone() = 0;

private:

    std::pair<size_t, size_t> fan_in_fan_out() {
        // this is basically stolen from pytorch.
        size_t num_input_fmaps = filters[0].size();
        size_t num_output_fmaps = filters.size();    
        size_t receptive_field_size = 1;
        if(aux::type_rank<Matrix>::value == 2) {
            // number of elements per filter.
            // since a filter is a row, it's the size of the row.
            receptive_field_size = filters[1].size();
        }
        size_t fan_in = num_input_fmaps * receptive_field_size;
        size_t fan_out = num_output_fmaps * receptive_field_size;
        return std::make_pair(fan_in, fan_out);
    }

protected:
    Matrix filters;

    size_t input_height;
    size_t input_width;
    size_t input_depth;

    ImageBatches last_input;
    ImageBatches last_output;

    std::string layer_type;

    double step_size;

    bool is_training = true;

};

#endif
