#include <vector>
#include <string>

#include "layers.hpp"

#ifndef DROPOUT_HPP
#define DROPOUT_HPP

template<typename Weight>
struct Dropout2d : public Layer_2D<Weight> {
    using Matrix = std::vector<std::vector<Weight>>;
    Dropout2d(std::string layer_name): Layer_2D<Weight>(layer_name) {} 

    Matrix forward_pass(const Matrix& input) {

    }

    Matrix operator()(const Matrix& input) {

    }

    Matrix backward_pass(const Matrix& input) {

    }
};

#endif
