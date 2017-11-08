
#include <vector>
#include <string>
#include <utility>
#include <thread>
#include <functional> //contains std::ref
#include <random>
#include <iostream>
#include <type_traits>

#include "net.hpp"
#include "data_manager.hpp"

#ifndef ENSEMBLE_HPP
#define ENSEMBLE_HPP


template<typename In, size_t InRank, typename Out, size_t OutRank,
         typename Weight = double>
class Ensemble {
public:
    using DataCont = std::vector<In>;
    using LabelCont = std::vector<Out>;

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
        manager(train_data,
                train_labels,
                test_data,
                test_labels,
                batch_size,
                ensemble_size),
        ensemble(ensemble_size, Net<In, InRank, Out, OutRank, Weight>(learning_rate, input)),
        ensemble_size(ensemble_size), 
        conv_data(cd),
        conv_label(cl)
    {
        conv_data = cd;
        conv_label = cl;
        if(batch_size % ensemble_size != 0) {
            std::cout << "batch size should be a multiple of ensemble size." << '\n';
            throw std::invalid_argument("bad batch and ensemble size");
        }
    }

    Net<In, InRank, Out, OutRank, Weight> get_net() {
        return ensemble[0];
    }
    
    void train(size_t epochs, bool verbose = false, size_t verbosity = 0) {
        for(size_t epoch = 0; epoch < epochs; ++epoch) {
            manager.process_all_data();
            if(epoch > 0) {
                manager.shuffle_data();
            }

            std::vector<std::thread> threads;
            
            for(size_t net = 0; net < ensemble_size; ++net) {
                threads.emplace_back(
                            &Ensemble::run_epoch,
                            std::ref(manager),
                            std::ref(ensemble[net]),
                            net);
            }

            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }

            average_ensemble();
        }
    }

    double test() {
        Net<In, InRank, Out, OutRank, Weight> test_net = get_net();

        auto test_pair = manager.test();

        int correct = 0;
        for(int datum = 0; datum < manager.test_size(); ++datum) {
            auto image = test_pair.first[datum];
            auto label = test_pair.second[datum];
            if(test_net.guess_and_check(image, label)) {
                correct += 1;
            }
        }
        return static_cast<double>(correct) / static_cast<double>(manager.test_size());
    }

private:

    Data_Manager<DataCont, LabelCont, Weight> manager; // holds test and train data

    std::vector<Net<In, InRank, Out, OutRank, Weight>> ensemble;
    size_t ensemble_size;
    std::vector<std::vector<double>> (*conv_data)(const DataCont&);
    std::vector<std::vector<double>> (*conv_label)(const LabelCont&);
  
    void average_ensemble() {
        for(int net = 1; net < ensemble_size; ++net) {
            ensemble[0] += ensemble[net];
        }
        ensemble[0] /= ensemble_size;
        for(int net = 1; net < ensemble_size; ++net) {
            ensemble[net] = ensemble[0];
        }
    }

    static void process_chunk(Net<In, InRank, Out, OutRank, Weight>& net, 
                              std::pair<DataCont, LabelCont>& chunk) {
        DataCont chunk_data = chunk.first;
        LabelCont chunk_labels = chunk.second;
        net.batch_update(chunk_data, chunk_labels); 
    }

    static void run_epoch(Data_Manager<DataCont, LabelCont, Weight>& manager, 
                          Net<In, InRank, Out, OutRank, Weight>& net, size_t net_id) 
    {
        for(size_t batch = 0; batch < manager.num_train_batches(); ++batch) {
            std::pair<DataCont, LabelCont> chunk = manager.get_chunk(net_id, batch);
            net.batch_update(chunk.first, chunk.second);
        } 
    }
};

#endif
