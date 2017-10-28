
#include <algorithm>
#include <vector>
#include <utility>
#include <functional>
#include <memory>
#include <iostream>
#include <string>
#include <map>
#include <boost/algorithm/string.hpp>

#include "layers.hpp"
#include "aux.hpp"
#include "dense.hpp"
#include "activation.hpp"
#include "conv2d.hpp"
#include "dropout.hpp"

#ifndef NET_HPP
#define NET_HPP

template<typename Weight = double>
class Net {
private:
    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;

    std::vector<std::unique_ptr<Layer_2D<Weight>>> layers;
    std::vector<std::unique_ptr<Layer_3D<Weight>>> layers_3d;

    Loss_Cross_Entropy<Weight> loss;

public:
    Net():loss() {}

    Net(Net&& other):
        layers(std::move(other.layers)),
        layers_3d(std::move(other.layers_3d)),
        loss(std::move(other.loss))
    {}

    Net& operator=(Net&& other) {
        if(this != &other) {
            layers = std::move(other.layers);
            layers_3d = std::move(other.layers_3d);
            loss = std::move(other.loss);
        }
        return *this;
    }

    // WILL NEED TO UPDATE FOR 3D layers.
    Net(const Net& other):
        layers(),
        layers_3d(other.layers_3d.size()),
        loss(other.loss)
    {
        for(auto& layer_ptr : other.layers) {
            std::unique_ptr<Layer_2D<Weight>> layer_ptr_clone(layer_ptr->clone());
            layers.push_back(std::move(layer_ptr_clone));
        }
    }

    Net(double learning_rate, const std::initializer_list<std::string>& input)
        {
            int tmp = 0;
            for(auto iter = input.begin(); iter != input.end(); ++iter) {
                ++tmp;
                std::string layer_string = *iter;
                std::vector<std::string> split_layer_string;
                boost::split(split_layer_string, layer_string,
                             boost::is_any_of("\t ,"));
                for(auto iter = split_layer_string.begin();
                    iter != split_layer_string.end()
                    ; ) // gross
                {
                    if(*iter == "\0") iter = split_layer_string.erase(iter);
                    else ++iter;
                }
                // construct dense ;)
                if(split_layer_string[0] == "dense") {
                    layers.push_back(
                        std::unique_ptr<Layer_2D<Weight>>(
                            new Dense<Weight>(std::stoi(split_layer_string[1]),
                                              std::stoi(split_layer_string[2]),
                                              learning_rate)
                        ));
                }
                // construct relu!
                else if(split_layer_string[0] == "relu") {
                    layers.push_back(
                        std::unique_ptr<Layer_2D<Weight>>(new Relu<Weight>()));
                }
                // construct the conv2d layer. Pretty gross, but w/e
                else if(split_layer_string[0] == "conv2d") {
                    layers_3d.push_back(
                        std::unique_ptr<Layer_3D<Weight>>(
                            new Conv2d<Weight>(
                                std::stoi(split_layer_string[1]),
                                std::stoi(split_layer_string[2]),
                                std::stoi(split_layer_string[3]),
                                std::stoi(split_layer_string[4]),
                                std::stoi(split_layer_string[5]),
                                std::stoi(split_layer_string[6]),
                                std::stoi(split_layer_string[7]),
                                learning_rate)));
                }
                else if(split_layer_string[0] == "dropout") {
                    layers.push_back(
                        std::unique_ptr<Layer_2D<Weight>>(
                            new Dropout2d<Weight>(
                                std::stod(split_layer_string[1], nullptr)
                            )));
                }
            }
        }

    void update(const Matrix& input, const Matrix& label) {
        Matrix&& prediction = predict(input, true);
        Matrix&& dloss = loss.comp_d_loss(prediction, label);
        Matrix& d_out = dloss;
        for(int layer = layers.size() - 1; layer >= 0; --layer) {
            d_out = layers[layer]->backward_pass(d_out);
        }
    }

    Matrix predict(Matrix input, bool training) {
        Matrix& trans = input;
        for(const auto& layer : layers) {
            layer->set_phase(training);
            trans = layer->forward_pass(trans);
        }
        return trans;
    }

    // returns true if guess was correct, else false.
    bool guess_and_check(Matrix input, Matrix label) {
        Matrix guess = predict(input, false);
        auto guess_iter = std::max_element(guess[0].begin(), guess[0].end());
        int guess_index = guess_iter - guess[0].begin();
        int label_index = 0;
        for( ; label[0][label_index] != 1; ++label_index) { }

        return guess_index == label_index;
    }
    /*
    // resizing for convolutions needs to be done here. 
    Image predict(Image input) {
        Image& trans_image = input;
        for(const auto& layer : layers) {
            Matrix&& column_trans = mat_aux::image_to_col(trans_image);
            column_trans = layer->forward_pass(column_trans);
            //trans_iamge = mat_aux::reshape(column_trans)
        }
    }*/
};


#endif
