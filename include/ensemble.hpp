
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
             std::initializer_list<std::string> input):
        train_data(train_data),
        train_labels(train_labels),
        test_data(test_data),
        test_labels(test_labels),
        ensemble(ensemble_size, Net<Weight>(learning_rate, input)), // SO ELEGANT UGHHGHH
        threads(ensemble_size),
        ensemble_size(ensemble_size), 
        data_order(train_labels.size()),
        batch_size(batch_size)
    {}

    Net<Weight> get_net() {
        /*
         * same as average ensemble, but return ensemble[0] instead of
         * making the rest equal ensemble[0]
         * need to implement move and copy constructors for nets.
         */
    }

    void train(size_t epochs, bool verbose = false) {
        for(int epoch = 0; epoch < epochs; ++epoch) {
            if(epoch > 0) {
                shuffle_data();
            }
            for(int d = 0; d < train_labels.size(); d += batch_size) {
                std::pair<DataCont, LabelCont> batch_pair = get_batch(); 
            }
        }
    }

    double test() {
        // test using the final net from the ensemble.
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
    
    std::pair<DataCont, LabelCont> get_batch() {
        // only for training data since you don't really need
        // batches for testing.
        
        if(batch_index > data_order.size()) {
            std::cout << "BATCH INDEX LARGER THAN AMOUNT OF DATA";
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
