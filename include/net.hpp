
#include <algorithm>
#include <vector>
#include <utility>
#include <functional>
#include <memory>
#include <iostream>
#include <string>
#include <map>
#include <boost/algorithm/string.hpp>
#include <type_traits>

#include "layers.hpp"
#include "aux.hpp"
#include "dense.hpp"
#include "relu.hpp"
#include "conv2d.hpp"
#include "dropout.hpp"

#ifndef NET_HPP
#define NET_HPP

template<typename In, size_t InRank, typename Out, size_t OutRank, typename Weight = double>
class Net {
private:
    //using Matrix = std::vector<std::vector<Weight>>;
    //using Image = std::vector<Matrix>;

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

    Net& operator=(const Net& other) {
        loss = other.loss;
        layers.resize(other.layers.size());
        layers_3d.resize(other.layers_3d.size());
        int index = 0;
        for(auto& layer_ptr : other.layers) {
            std::unique_ptr<Layer_2D<Weight>> layer_ptr_clone(layer_ptr->clone());
            layers[index] = std::move(layer_ptr_clone);
            ++index;
        }
        for(auto& layer_ptr : other.layers_3d) {
            std::unique_ptr<Layer_3D<Weight>> layer_ptr_clone(layer_ptr->clone());
            layers_3d[index] = std::move(layer_ptr_clone);
            ++index;
        }
        return *this;
    }

    // WILL NEED TO UPDATE FOR 3D layers.
    Net(const Net& other):
        layers(),
        layers_3d(),
        loss(other.loss)
    {
        for(auto& layer_ptr : other.layers) {
            std::unique_ptr<Layer_2D<Weight>> layer_ptr_clone(layer_ptr->clone());
            layers.push_back(std::move(layer_ptr_clone));
        }
        for(auto& layer_ptr : other.layers_3d) {
            std::unique_ptr<Layer_3D<Weight>> layer_ptr_clone(layer_ptr->clone());
            layers_3d.push_back(std::move(layer_ptr_clone));
        }
    }

    Net(double learning_rate, const std::initializer_list<std::string>& input)
        {
            size_t ordering_index = 0;
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
                ++ordering_index;
            }
        }


    void update(const In& input, const Out& label) {
        Out&& prediction = predict(input, true);
        Out&& dloss = loss.comp_d_loss(prediction, label);
        Out& d_out = dloss;
        for(int layer = layers.size() - 1; layer >= 0; --layer) {
            d_out = layers[layer]->backward_pass(d_out);
        }
    }

    void batch_update(const std::vector<In>& inputs, 
                      const std::vector<Out>& labels) 
    {
        const size_t batch_size = inputs.size();
        std::vector<Out> predictions(batch_size);
 
        for(int i = 0; i < batch_size; ++i) {
            predictions[i] = predict(inputs[i], true);
        }
        
        std::vector<Out> losses(batch_size);
        for(int i = 0; i < batch_size; ++i) {
            losses[i] = loss.comp_d_loss(predictions[i], labels[i]);
        } 
        
        Out d_out = losses[0];
        for(int i = 1; i < losses.size(); ++i) {
            d_out = aux::matadd(losses[0], losses[i]);
        }

        for(auto& row : d_out) {
            for(auto& item : row) {
                item /= batch_size;
            }
        }

        for(int layer = layers.size() - 1; layer >= 0; --layer) {
            d_out = layers[layer]->backward_pass(d_out);
        }
        // implement unflatten for updating 3d layers.

    }

    // need to do something about intermediate variables not being In or Out...

    Out predict(In input, bool training) {
        if constexpr (InRank == 3) {
            return predict_2d(aux::flatten_3d(predict_3d(input, training)), training); 
        }
        
        else if constexpr (InRank == 2)  
            return predict_2d(input, training); 
        else {
            std::cout << "Input dimensions are invalid. Predict(In, bool)\n";
        }
    }

    In predict_3d(In input, bool training) {
        for(const auto& layer : layers_3d) {
            layer->set_phase(training);
            input = layer->forward_pass(input);
        }
        return input;
    }

    Out predict_2d(Out input, bool training) {
        Out& trans = input;
        for(const auto& layer : layers) {
            layer->set_phase(training);
            trans = layer->forward_pass(trans);
        }
        return trans;
    }

    template<typename InputType, typename LayerType>
    auto process_layer(LayerType& layer, const InputType& layer_input, bool training) 
        -> decltype(layer->forward_pass(layer_input)) 
    {
        return layer->forward_pass(layer_input);
    }

    // returns true if guess was correct, else false.
    bool guess_and_check(In input, Out label) {
        Out guess = predict(input, false);
        auto guess_iter = std::max_element(guess[0].begin(), guess[0].end());
        int guess_index = guess_iter - guess[0].begin();
        auto label_iter = std::max_element(label[0].begin(), label[0].end());
        int label_index = label_iter - label[0].begin();
    
        return guess_index == label_index;
    }
    
    Net<In, InRank, Out, OutRank, Weight>& 
    operator+=(const Net<In, InRank, Out, OutRank, Weight>& other) 
    {
        for(int layer = 0; layer < layers.size(); ++layer) {
            *(layers[layer]) += *(other.layers[layer]);
        }
        return *this;
    }

    Net<In, InRank, Out, OutRank, Weight>& operator/=(const size_t count) {
        for(int layer = 0; layer < layers.size(); ++layer) {
            *(layers[layer]) /= count;
        }  
        return *this;
    }
};


#endif
