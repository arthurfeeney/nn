#include <algorithm>
#include <vector>
#include <utility>
#include <string>
#include <tuple>
#include <memory>

#include "aux.hpp"
#include "layers.hpp"

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

template <typename Weight = double>
class Relu : public Layer_2D<Weight> {
public:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    Relu(): Layer_2D<Weight>("relu") {}

    Relu(Relu&& other):Layer_2D<Weight>(std::move(other)) {}

    Relu(const Relu& other): Layer_2D<Weight>(other) {}

    Relu* clone() {
        return new Relu(*this);
    }

    Matrix forward_pass(const Matrix& input) {
        this->last_input = input;
        return relu(input);
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
private:
    Matrix relu(const Matrix& c) { 
        Matrix relud_c(c.size(), std::vector<Weight>(c[0].size()));
        for(int i = 0; i < c.size(); ++i) {
            for(int j = 0; j < c[0].size(); ++j) {
                relud_c[i][j] = std::max<double>(c[i][j], 0);
            }
        }
        return relud_c;
    }

    Image relu(const Image& c) {
        Image relud_c(c.size(), 
                      std::vector<Weight>(c[0].size(), 
                                          std::vector<Weight>(c[0][0].size())));
        for(int i = 0; i < c.size(); ++i) {
            for(int j = 0; j < c[0].size(); ++j) {
                for(int k = 0; k < c[0][0].size(); ++k) {
                    relud_c[i][j][k] = std::max<double>(c[i][j][k], 0);
                }
            }
        }
    }
};

#endif
