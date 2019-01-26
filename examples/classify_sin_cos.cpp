
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <cmath>
#include <algorithm>

#include "../include/ensemble.hpp"
#include "../include/Optimizer/momentum.hpp"
#include "../include/Optimizer/adam.hpp"
#include "../include/Initializer/xavier_normal.hpp"


constexpr double pi = 3.1415926525;

using Dataset = std::vector<std::vector<std::vector<double>>>;



template<typename RealType = double>
class UniformRand {
private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<RealType> d;
public:
    UniformRand(RealType mean = 0.0, RealType var = 1.0):
        rd(), gen(rd()), d(mean, var) {}

    RealType operator()() {
        return d(gen);
    }
};

template<typename RealType>
auto rand_sin(UniformRand<RealType>& g) {
    RealType n = g() * pi;
    return std::vector { std::vector {
        n, std::sin(n)
    }};
}

template<typename RealType>
auto rand_cos(UniformRand<RealType>& g) {
    RealType n = g() * pi;
    return std::vector { std::vector {
        n, std::cos(n)
    }};
}

std::pair<Dataset, Dataset> generate_data(int size) {
    UniformRand<double> g(0, 2);

    Dataset sin_data(size / 2);
    Dataset cos_data(size / 2);
    for(auto& datum : sin_data) 
        datum = rand_sin(g);
    for(auto& datum : cos_data) 
        datum = rand_cos(g);

    Dataset labels(size);
    for(int i = 0; i < size / 2; ++i) {
        labels[i] = std::vector {std::vector {1.0, 0.0}};
    }

    for(int i = size / 2; i < size; ++i) {
        labels[i] = std::vector {std::vector {0.0, 1.0}};
    }
    // concatenate sin and cos data
    sin_data.insert(sin_data.end(),
                    cos_data.begin(),
                    cos_data.end());
    Dataset data = sin_data;

    // reorder the train data, applying the same shuffle to inputs and labels
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_shuffle(indices.begin(), indices.end());

    for(int i = 0; i < size; ++i) {
        data[i] = data[indices[i]];
        labels[i] = labels[indices[i]];
    }

    return std::make_pair(data, labels);
}

int main() {

    int train_size = 1000;
    int test_size = 1000;

    auto train = generate_data(train_size);
    auto test = generate_data(test_size);

    Dataset train_data = train.first;
    Dataset train_labels = train.second;

    Dataset test_data = test.first;
    Dataset test_labels = test.second;

    Ensemble<std::vector<std::vector<double>>, 
             std::vector<std::vector<double>>,
             Adam<>,
             double>
    net(
        train_data,
        train_labels,
        test_data, 
        test_labels,
        1, // Number of networks in the ensemble.
        1e-3, // learning rate.
        5000, // num steps between lr decay
        1, // amount to decay by. (No decay)
        1,
        {
            "dense 20 2",
            "relu",
            "dense 20 20",
            "relu",
            "dense 20 20",
            "relu",
            "dense 2 20"
        }
     );
     
    net.initialize(init::xavier_normal<std::vector<std::vector<double>>, double>, 1);

    net.train(10, true, 100);

    double test_acc = net.test();

    std::cout << test_acc << '\n';




    return 0;
}
