
#include <vector>
#include <string>
#include <utility>
#include <thread>
#include <functional> //contains std::ref
#include <random>
#include <iostream>

#include "net.hpp"

#ifndef ENSEMBLE_HPP
#define ENSEMBLE_HPP


template<typename DataCont, typename LabelCont, typename Weight = double>
class Ensemble {
public:
    Ensemble(DataCont train_data,
             LabelCont train_labels,
             DataCont test_data,
             LabelCont test_labels,
             size_t ensemble_size,
             double learning_rate, 
             size_t batch_size,
             std::initializer_list<std::string> input,
             std::vector<std::vector<double>> (*cd)(const DataCont&) = nullptr, 
             std::vector<std::vector<double>> (*cl)(const LabelCont&) = nullptr):
        train_data(train_data),
        train_labels(train_labels),
        test_data(test_data),
        test_labels(test_labels),
        ensemble(ensemble_size, Net<Weight>(learning_rate, input)), // SO ELEGANT UGHHGHH
        threads(ensemble_size),
        ensemble_size(ensemble_size), 
        data_order(train_labels.size()),
        batch_size(batch_size),
        conv_data(cd),
        conv_label(cl)
    {
        //conv_data = cd;
        //conv_label = cl;
        if(batch_size % ensemble_size != 0) {
            std::cout << "batch size should be a multiple of ensemble size." << '\n';
            throw std::invalid_argument("bad batch and ensemble size");
        }
        std::iota(data_order.begin(), data_order.end(), 0);
    }

    Net<Weight> get_net() {
        return ensemble[0];
    }

    void train(size_t epochs, bool verbose = false, size_t verbosity = 0) {
        for(size_t epoch = 0; epoch < epochs; ++epoch) {
            if(epoch > 0) {
                shuffle_data();
            }
            for(size_t step = 0; step < train_labels.size(); step += batch_size) {
                if(step % verbosity == 0) {
                    std::cout << "epoch: " << epoch << ", step: " << step << '\n';
                }

                std::pair<DataCont, LabelCont> batch_pair = get_batch(); 
                auto each_net_chunk = chunk_batch(batch_pair); 
            
                for(size_t net = 0; net < ensemble_size; ++net) {
                    DataCont chunk_data = each_net_chunk[net].first;
                    LabelCont chunk_labels = each_net_chunk[net].second;
                    for(size_t datum = 0; datum < chunk_labels.size(); ++datum) {
                        ensemble[net].update(chunk_data[datum],
                                             chunk_labels[datum]);
                    }
                }
            }
        }
    }

    double test() {
        Net<Weight> test_net = get_net();

        int correct = 0;
        for(int datum = 0; datum < test_labels.size(); ++datum) {
            auto image = test_data[datum];
            auto label = test_labels[datum];
            if(test_net.guess_and_check(image, label)) {
                correct += 1;
            }
        }
        return static_cast<double>(correct) / static_cast<double>(test_labels.size());
    }

private:
    DataCont train_data;
    LabelCont train_labels;
    DataCont test_data;
    LabelCont test_labels;

    std::vector<Net<Weight>> ensemble;
    std::vector<std::thread> threads;
    size_t ensemble_size;
    
    std::vector<int> data_order; // stores indices for getting batches and shit.
    size_t batch_index = 0;
    size_t batch_size;

    std::vector<std::vector<double>> (*conv_data)(const DataCont&);
    std::vector<std::vector<double>> (*conv_label)(const LabelCont&);
    
    std::pair<DataCont, LabelCont> get_batch() {
        // only for training data since you don't really need
        // batches for testing.
        
        if(batch_index + batch_size >= data_order.size()) {
            batch_index = 0;
        }
        
        std::vector<int> indices(data_order.begin() + batch_index, 
                                 data_order.begin() + batch_index + batch_size);

        DataCont data_batch(batch_size);
        LabelCont label_batch(batch_size);
        
        for(int k = 0; k < batch_size; ++k) {
            data_batch[k] = train_data[indices[k]];
            label_batch[k] = train_labels[indices[k]];
        }

        batch_index += batch_size;

        return std::make_pair(data_batch, label_batch);
    }

    std::vector<std::pair<DataCont, LabelCont>> 
    chunk_batch(const std::pair<DataCont, LabelCont>& batch) {
        DataCont data = batch.first;
        LabelCont labels = batch.second;
        size_t chunk_size = batch_size / ensemble_size;
        std::vector<std::pair<DataCont, LabelCont>> chunks(ensemble_size);
        for(size_t i = 0, index = 0; 
            i < batch_size; 
            i += chunk_size, ++index) 
        {
            DataCont d_c(data.begin() + i, data.begin() + i + chunk_size);
            LabelCont l_c(labels.begin() + i, labels.begin() + i + chunk_size);
            chunks[index] = std::make_pair(d_c, l_c);
        }
        return chunks;
    }

    void shuffle_data() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data_order.begin(), data_order.end(), g);
    }

    void average_ensemble() {
        /*
         * ensemble[0] = sum(ensemble) / ensemble_size
         * for i in 1 until ensemble_size:
         *  ensemble[i] = ensemble[0]
         *  so ez lmao
         */
    }

    void init_ensemble() {
         
    }


};

#endif
