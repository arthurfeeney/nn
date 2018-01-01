
#ifndef ENSEMBLE_HPP
#define ENSEMBLE_HPP

#include <vector>
#include <string>
#include <utility>
#include <thread>
#include <functional>
#include <random>
#include <iostream>
#include <type_traits>
#include <cmath>

#include "net.hpp"
#include "data_manager.hpp"
#include "Optimizer/SGD.hpp"

template<typename In, typename Out, typename Opt = SGD, 
         typename Weight = double>
class Ensemble {
public:
    using DataCont = std::vector<In>;
    using LabelCont = std::vector<Out>;

    using InRank = aux::type_rank<In>;
    using OutRank = aux::type_rank<Out>;

    Ensemble(DataCont train_data,
             LabelCont train_labels,
             DataCont test_data,
             LabelCont test_labels,
             size_t ensemble_size,
             double learning_rate, 
             size_t batch_size,
             std::initializer_list<std::string> input,
             size_t n_threads = 1, // num of threads available to EACH network.
             size_t validation_set_size = 0,
             In (*cd)(const DataCont&) = nullptr, 
             Out (*cl)(const LabelCont&) = nullptr):
        manager(train_data,
                train_labels,
                test_data,
                test_labels,
                batch_size,
                ensemble_size,
                validation_set_size),
        ensemble(ensemble_size, 
                 Net<In, 
                     aux::type_rank<In>::value, 
                     Out, 
                     aux::type_rank<Out>::value,
                     Opt,
                     Weight>(learning_rate, input)),
        ensemble_size(ensemble_size), 
        n_threads(n_threads),
        conv_data(cd),
        conv_label(cl)
    {
        conv_data = cd;
        conv_label = cl;
        if(batch_size % ensemble_size != 0) {
            std::cout << "batch size should be a multiple of ensemble size." 
                      << '\n';
            throw std::invalid_argument("bad batch and ensemble size");
        }
    }

    Net<In, InRank::value, Out, OutRank::value, Opt, Weight>
    get_test_net() 
    {
        return ensemble[0];
    }
    
    void train(size_t epochs, bool verbose = false, size_t verbosity = 0) {

        std::vector<std::thread> threads;

        for(size_t net = 0; net < ensemble_size; ++net) {
            threads.emplace_back(&Ensemble::run_train_cycle,
                                 std::ref(manager),
                                 std::ref(ensemble[net]),
                                 net,
                                 epochs,
                                 verbosity,
                                 n_threads);
        }

        for(auto& thread : threads) {
            thread.join();
        }

        average_ensemble();
    }

    void async_train_variant(size_t epochs, bool verbose = false, 
                             size_t verbosity = 0) 
    {
        std::vector<double> prev_error(3, 1); // used for validation
        for(size_t epoch = 0; epoch < epochs; ++epoch) {

            manager.process_all_data();

            if(epoch > 0) {
                manager.shuffle_data();
            }

            std::vector<std::thread> threads;
            // train each network on its own cluster of threads for one epoch.
            for(size_t net = 0; net < ensemble_size; ++net) {
                threads.emplace_back(
                            &Ensemble::run_epoch,
                            std::ref(manager),
                            std::ref(ensemble[net]),
                            net,
                            epoch,
                            verbosity,
                            n_threads);
            }

            for(auto& thread : threads) {
                if(thread.joinable()) {
                    thread.join();
                }
            }
            
            average_ensemble();
            
            if(manager.validation_size() > 0) {
                // ONLY USED IF THERE IS A VALIDATION SET.
                // validate modifies prev_error.
                // if the change in loss is within threshold, stop training 
                if(!validate(0.00001, prev_error)) break; 
            }
        }
    }

    // tests the quality of the network on the test set.
    // returns the percent it got correct.
    double test() {
        Net<In, InRank::value, Out, OutRank::value, Opt, Weight> 
            test_net = get_test_net();

        auto test_pair = manager.test();

        int correct = 0;
        for(size_t datum = 0; datum < manager.test_size(); ++datum) {
            auto image = test_pair.first[datum];
            auto label = test_pair.second[datum];
            if(test_net.guess_and_check(image, label, n_threads)) {
                correct += 1;
            }
        }
        return static_cast<double>(correct) / 
               static_cast<double>(manager.test_size());
    }

private:

    Data_Manager<DataCont, LabelCont, Weight> manager; // holds data

    std::vector<Net<In, InRank::value, Out, OutRank::value, Opt, Weight>>
        ensemble;

    size_t ensemble_size;

    size_t n_threads; // number of threads available to EACH network.
    
    // optional functions to convert data to needed format.
    In (*conv_data)(const DataCont&);
    Out (*conv_label)(const LabelCont&);
  
    void average_ensemble() {
        for(size_t net = 1; net < ensemble_size; ++net) {
            ensemble[0] += ensemble[net];
        }
        ensemble[0] /= ensemble_size;
        for(size_t net = 1; net < ensemble_size; ++net) {
            ensemble[net] = ensemble[0];
        }
    }

    static void 
    process_chunk(Net<In, InRank::value, 
                      Out, OutRank::value, 
                      Opt, Weight
                  >& net, 
                  std::pair<DataCont, LabelCont>& chunk)
    {
        DataCont chunk_data = chunk.first;
        LabelCont chunk_labels = chunk.second;
        net.batch_update(chunk_data, chunk_labels); 
    }
    
    static void
    run_train_cycle(Data_Manager<DataCont, LabelCont, Weight>& manager,
                    Net<In, InRank::value, Out, OutRank::value, Opt, Weight>&
                    net,
                    size_t net_id, size_t epochs, size_t verbosity = 0,
                    const size_t num_threads = 1)
    {
        for(size_t epoch = 0; epoch < epochs; ++epoch) {
            manager.process_all_data();
            if(epoch > 0) {
                manager.shuffle_data();
            }
            run_epoch(manager, net, net_id, epoch, verbosity, num_threads);
        }
    }



    // trains a network on the entire dataset one time.
    static void 
    run_epoch(Data_Manager<DataCont, LabelCont, Weight>& manager, 
              Net<In, InRank::value, Out, OutRank::value, Opt, Weight>& net,
              size_t net_id, unsigned int epoch, unsigned int verbosity = 0,
              const size_t num_threads = 1) 
    {
        for(size_t batch = 0; batch < manager.num_train_batches(); ++batch) {
            if(net_id == 0 && verbosity && batch % verbosity == 0) {
                // only the 'main' net prints
                std::cout << "epoch: " << epoch << " step: " << 
                             batch * manager.step_size() << '\n';
            }
            std::pair<DataCont, LabelCont> chunk = 
                manager.get_chunk(net_id, batch);
            net.batch_update(chunk.first, chunk.second, num_threads);
        } 
    }

    bool validate(double threshold, std::vector<double>& prev_error) {
        // if validation error is greater than threshold, retern false.
        // other wise return true
        // if false, training will stop.
        Net<In, InRank::value, Out, OutRank::value, Opt, Weight> 
            validation_net = get_test_net();
        std::pair<DataCont, LabelCont> 
            validation_set = manager.validation();
        DataCont& validation_data = validation_set.first;
        LabelCont& validation_labels = validation_set.second;

        double total_error = 0;

        for(size_t i = 0; i < validation_labels.size(); ++i) {
            total_error += validation_net.comp_loss(validation_data[i],
                                                    validation_labels[i]);
        }

        double new_error = total_error / validation_labels.size();
        std::cout << "validation loss: " << new_error << '\n';

        // prev_error[0] is the most recent old error !!!!!!

        // if the new error is greater than all the last three errors, stop!
        bool should_continue = false;
        for(double val : prev_error) {
            if(new_error <= val) {
                should_continue = true;
            }
        }

        // if the change in error is not in the threshold, it should 
        // continue.
        //bool valid = threshold < std::abs(prev_error[0] - new_error);

        // scooch everything over and put the new error in the list.
        prev_error[2] = prev_error[1];
        prev_error[1] = prev_error[0];
        prev_error[0] = new_error;

        // if it is not in the threshold and not increasing, continue.
        return should_continue;//valid && should_continue;
    }
};

#endif
