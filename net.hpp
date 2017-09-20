
#include <algorithm>
#include <vector>
#include <utility>
#include <functional>
#include <memory>
#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>

#include "layers.hpp"
#include "aux.hpp"

#ifndef NET_HPP
#define NET_HPP

template<typename Weight = double>
class Net {
private:

    using Matrix = std::vector<std::vector<Weight>>;
    using Image = std::vector<Matrix>;


    std::vector<std::unique_ptr<Layer<Weight>>> layers;
    Loss_Cross_Entropy<Weight> loss;
    
public:
    Net():loss() {}

    Net(const std::initializer_list<std::string>& input):
        layers(input.size())
        {
            for(auto iter = input.begin(); iter != input.end(); ++iter) {
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
                if(split_layer_string[0] == "dense") {
                    layers[iter - input.begin()] = 
                        std::unique_ptr<Layer<Weight>>(
                            new Dense<Weight>(std::stoi(split_layer_string[1]), 
                                              std::stoi(split_layer_string[2]))
                        );
                }

                else if(split_layer_string[0] == "relu") {
                    layers[iter - input.begin()] = 
                        std::unique_ptr<Layer<Weight>>(new Relu<Weight>());
                }
/*
                else if(split_layer_string[0] == "conv2d") {

                    layers[iter - input.begin()] =
                        std::unique_ptr<Layer<Weight>>(
                            new Conv2d<Weight>(
                                std::stoi(split_layer_string[1]),
                                std::stoi(split_layer_string[2]),
                                std::stoi(split_layer_string[3]),

                            )
                        );
                }*/
            }            
        }

    template<typename S>
    void update(const S& input, const S& label) {
        Matrix&& prediction = predict(input);
        Matrix&& dloss = loss.comp_d_loss(prediction, label);
        Matrix& d_out = dloss;
        for(int layer = layers.size() - 1; layer >= 0; --layer) {
            d_out = layers[layer]->backward_pass(d_out);
        }
    }

    Matrix predict(Matrix input) {
        Matrix& trans = input;
        for(const auto& layer : layers) {
            trans = layer->forward_pass(trans);
        }
        return trans;
    }

    // need to flatten all images to matrix, before input. 
    Image predict(Image input) {
        Image& trans_image = input;
        for(const auto& layer : layers) {
            Matrix&& column_trans = mat_aux::image_to_col(trans_image);
            column_trans = layer->forward_pass(column_trans);
            //trans_iamge = mat_aux::reshape(column_trans)
        }
    }
    
    Layer<Weight> operator[](int index) {
        // returns layer at input index;
        return *layers[index];
    }
};


#endif 
