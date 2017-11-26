
/*
* in aux.hpp.
* A loss function. Probably need to make some better ones to use...
*/

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <utility>
#include <iostream>
#include <type_traits>

#include "aux.hpp"

#ifndef LOSS_HPP
#define LOSS_HPP

template<typename P, typename S>
static double correct(P probs, S label) {
    /*
    * label is one hot encoding.
    * returns prob for the correct class.
    * throws an error if it wasn't found.
    */
    for(size_t index = 0; index < label.size(); ++index) {
        if(label[index] == 1) {
            return probs[index];
        }
    }
    throw("1 was not found, no correct answer to return?");
}

template<typename S>
static int correct_index(S label) {
    for(size_t index = 0; index < label.size(); ++index) {
        if(label[index] == 1) {
            return index;
        }
    }
    throw("1 was not found in one hot encoding.");
}

using S = std::vector<std::vector<double>>;

std::pair<S, double> loss(S scores, 
                         S actual) {
    /*
    * actual is a batch of one hot encodings.
    * Whichever index contains 1, is correct.
    * probs is a batch of probabilities for each class.
    */
    size_t num_examples = scores[0].size();
    S exp_scores = aux::exp(scores);
    std::vector<double> sums(0);
    for(const auto& row : exp_scores) {
        sums.push_back(std::accumulate(row.begin(), row.end(), 0.0));
    }

    S probs(exp_scores);


    for(size_t row = 0; row < probs.size(); ++row) {
        for(size_t col = 0; col < probs[0].size(); ++col) {
            probs[row][col] /= sums[row];
        }
    }

    std::vector<double> correct_logprobs(scores.size(), 0);

    for(size_t row = 0; row < probs.size(); ++row) {
        correct_logprobs[row] = correct(probs[row], actual[row]);
    }

    correct_logprobs = aux::log(correct_logprobs);

    for(auto& item : correct_logprobs) {
        item = -item;
    }

    // sum of correct_logprobs divided by num_examples.
    double add_probs = std::accumulate(correct_logprobs.begin(),
                                       correct_logprobs.end(),
                                       0.0);

    double data_loss = add_probs / num_examples;

    // I don't think I am going to use regularization loss... For now.

    double loss = data_loss;

    return make_pair(probs, loss);
}

template<typename P, typename S>
P dloss(P probs, S actual) {
    size_t num_examples = probs.size();
    P dscores = probs;
    for(size_t row = 0; row < probs.size(); ++row) {
        size_t col = correct_index(actual[row]);
        dscores[row][col] -= 1;
    }
    for(auto& row : dscores) {
        for(auto& item : row) {
            item /= num_examples;
        }
    }
    return dscores;
}

#endif
