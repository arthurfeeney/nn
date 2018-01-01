
#ifndef NET_HPP
#define NET_HPP

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
#include "lrelu.hpp"
#include "prelu.hpp"
#include "tanh.hpp"
#include "sigmoid.hpp"
#include "batch_normalization.hpp"
#include "softplus.hpp"

template<typename In, size_t in_rank, typename Out, size_t out_rank,
         typename Opt, typename Weight = double>
class Net {
private:

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
            std::unique_ptr<Layer_2D<Weight>> 
                layer_ptr_clone(layer_ptr->clone());
            
            layers[index] = std::move(layer_ptr_clone);
            ++index;
        }
        index = 0;
        for(auto& layer_ptr : other.layers_3d) {
            std::unique_ptr<Layer_3D<Weight>> 
                layer_ptr_clone(layer_ptr->clone());
            layers_3d[index] = std::move(layer_ptr_clone);
            ++index;
        }
        return *this;
    }

    Net(const Net& other):
        layers(),
        layers_3d(),
        loss(other.loss)
    {
        for(auto& layer_ptr : other.layers) {
            std::unique_ptr<Layer_2D<Weight>> 
                layer_ptr_clone(layer_ptr->clone());
            
            layers.push_back(std::move(layer_ptr_clone));
        }
        for(auto& layer_ptr : other.layers_3d) {
            std::unique_ptr<Layer_3D<Weight>> 
                layer_ptr_clone(layer_ptr->clone());
            
            layers_3d.push_back(std::move(layer_ptr_clone));
        }
    }

    // construct network given input string and learning rate.
    // parses input to to generate layers and put in the layers vector.
    Net(double learning_rate, const std::initializer_list<std::string>& input)
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
            // construct dense ;)
            if(split_layer_string[0] == "dense") {
                layers.push_back(
                    std::unique_ptr<Layer_2D<Weight>>(
                        new Dense<Opt, Weight>(
                                          std::stoi(split_layer_string[1]),
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
            // constructs leaky ReLU.
            else if(split_layer_string[0] == "leaky" || 
                    split_layer_string[0] == "lrelu") 
            {
                // if scale is not specified, it defaults to 0.01.
                if(split_layer_string.size() == 1) {
                    layers.push_back(
                        std::unique_ptr<Layer_2D<Weight>>(
                            new LRelu<Weight>()
                        )
                    );
                }
                // constructs the layer with the scale specified. 
                else {
                    double scale = std::stod(split_layer_string[1], nullptr);
                    layers.push_back(
                        std::unique_ptr<Layer_2D<Weight>>(
                            new LRelu<Weight>(scale)
                        )
                    );
                }
            }
            else if(split_layer_string[0] == "prelu")
            {
                size_t layer_size = std::stoi(split_layer_string[1]);
                layers.push_back(
                    std::unique_ptr<Layer_2D<Weight>>(
                        new PRelu<Opt, Weight>(layer_size, learning_rate)
                    )
                );
            }
            else if(split_layer_string[0] == "tanh")
            {
                layers.push_back(
                    std::unique_ptr<Layer_2D<Weight>>(
                        new Tanh<Weight>()
                    )
                );
            }
            else if(split_layer_string[0] == "sigmoid")
            {
                layers.push_back(
                    std::unique_ptr<Layer_2D<Weight>>(
                        new Sigmoid<Weight>()
                    )
                );
            }
            else if(split_layer_string[0] == "batchnorm" ||
                    split_layer_string[0] == "bn")
            {
                double eps = std::stod(split_layer_string[1], nullptr);
                double mom = std::stod(split_layer_string[2], nullptr);
                layers.push_back(
                    std::unique_ptr<Layer_2D<Weight>>(
                        new BatchNorm<Opt, Weight>(eps, mom)
                    )
                );
            }
            else if(split_layer_string[0] == "softplus")
            {
                layers.push_back(
                    std::unique_ptr<Layer_2D<Weight>>(
                        new Softplus<Weight>()
                    )
                );
            }
        }
    }

    // updates the network for a single input.
    void update(const In& input, const Out& label) {
        Out&& prediction = predict(input, true);
        Out&& dloss = loss.comp_d_loss(prediction, label);
        Out& d_out = dloss;
        for(int layer = layers.size() - 1; layer >= 0; --layer) {
            d_out = layers[layer]->backward_pass(d_out);
        }
    }

    // updates the network given a batch of inputs (possibly only 1).
    void batch_update(const std::vector<In>& inputs, 
                      const std::vector<Out>& labels,
                      const size_t n_threads = 1) 
    {
        // compute the loss.
        const size_t batch_size = inputs.size();

        In input(batch_size);
        Out label(batch_size);
        for(size_t i = 0; i < batch_size; ++i) {
            input[i] = inputs[i][0];
            label[i] = labels[i][0];
        }

        Out prediction = predict(input, true, n_threads);

        Out d_out(loss.comp_d_loss(prediction, label));
         
        for(auto& row : d_out) {
            for(auto& val : row) {
                val /= batch_size;
            }
        }
        
        // apply backward pass to each layer iteratively to update.
        for(int layer = layers.size() - 1; layer >= 0; --layer) {
            d_out = n_threads > 1 ? 
                        layers[layer]->async_backward_pass(d_out, n_threads) :
                        layers[layer]->backward_pass(d_out);        
        }

        // update 3d layers if they exist.
        if constexpr(in_rank == 3) {
            std::tuple<int, int, int> dims = 
                layers_3d[layers_3d.size() - 1]->proper_output_dim();

            int height = std::get<0>(dims);
            int width = std::get<1>(dims);
            int depth = std::get<2>(dims);

            In d_out_3d = aux::unflatten(d_out, depth, height, width);

            for(int layer = layers_3d.size() - 1; layer >= 0; --layer) {
                d_out_3d = layers_3d[layer]->backward_pass(d_out_3d);
            }
        }
    }

    // compute output for given input.
    Out predict(In input, bool training, const size_t n_threads) {
        // if 3d layers exist, process them first.
        if constexpr (in_rank == 3) {
            auto prediction_3d = predict_3d(input, training, n_threads);
            auto flattened = aux::flatten_3d(prediction_3d);
            return predict_2d(flattened, training, n_threads); 
        } 
        else if constexpr (in_rank == 2)  
            return predict_2d(input, training, n_threads); 
        else {
            std::cout << "Input dimensions are invalid. Predict(In, bool)\n";
        }
        return Out();
    }

    In predict_3d(In input, bool training, const size_t n_threads) {
        for(const auto& layer : layers_3d) {
            layer->set_phase(training);
            input = layer->forward_pass(input);
        }
        return input;
    }

    Out predict_2d(Out input, bool training, const size_t n_threads) {
        Out& trans = input;
        for(const auto& layer : layers) {
            layer->set_phase(training);
            trans = n_threads > 1 ?
                        layer->async_forward_pass(input, n_threads) :
                        layer->forward_pass(trans);
        }
        return trans;
    }

    template<typename InputType, typename LayerType>
    auto process_layer(LayerType& layer, 
                       const InputType& layer_input, 
                       bool training) 
    {
        return layer->forward_pass(layer_input);
    }

    // returns true if guess was correct, else false.
    bool guess_and_check(In input, Out label, size_t n_threads = 1) {
        Out guess = predict(input, false, n_threads);
        auto guess_iter = std::max_element(guess[0].begin(), guess[0].end());
        size_t guess_index = guess_iter - guess[0].begin();
        auto label_iter = std::max_element(label[0].begin(), label[0].end());
        size_t label_index = label_iter - label[0].begin();
    
        return guess_index == label_index;
    }

    double comp_loss(In input, Out label, size_t n_threads = 1) {
        Out guess = predict(input, false, n_threads);
        loss.forward_pass(guess, label);
        return loss.get_loss();
    }
    
    // used for averaging the networks. element-wise sum of all weights.
    Net<In, in_rank, Out, out_rank, Opt, Weight>& 
    operator+=(const Net<In, in_rank, Out, out_rank, Opt, Weight>& other) 
    {
        for(size_t layer = 0; layer < layers.size(); ++layer) {
            *(layers[layer]) += *(other.layers[layer]);
        }
        for(size_t layer = 0; layer < layers_3d.size(); ++layer) {
            *(layers_3d[layer]) += *(other.layers_3d[layer]);
        }
        return *this;
    }

    // used for averaging the networks. divides all weights by scalar.
    Net<In, in_rank, Out, out_rank, Opt, Weight>& 
    operator/=(const size_t scalar) 
    {
        for(size_t layer = 0; layer < layers.size(); ++layer) {
            *(layers[layer]) /= scalar;
        }
        for(size_t layer = 0; layer < layers_3d.size(); ++layer) {
            *(layers_3d[layer]) /= scalar;
        } 
        return *this;
    }

    // adds up every weight in the network.
    double sum_all_weights() const {
        double sum = 0;
        for(auto& layer : layers) {
            for(auto& row : layer->get_weights()) {
                for(auto& val : row) {
                    sum += val;
                }
            }
        }
        // TODO: implement for 3d layers.
        return sum;
    }
};


#endif
