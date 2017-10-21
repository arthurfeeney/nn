#include <iostream>
#include <vector>
#include <memory>
#include <utility>

#include "aux.hpp"
#include "layers.hpp"
#include "net.hpp"
#include "mnist_data/mnist/include/mnist/mnist_reader.hpp"


using std::vector;
using std::make_unique;
using std::pair;
using std::make_pair;

using Matrix = vector<vector<double>>;
using Image = vector<Matrix>;

void print(vector<vector<double>>& m) {
    for(auto& row : m) {
        for(auto& item : row) {
            std::cout << item << ' ';
        }
        std::cout << '\n';
    }
}

pair<vector<Matrix>, vector<Matrix>> generate_data(int size) {
    vector<Matrix> random_data(size, Matrix{{0}});
    vector<Matrix> labels(size, Matrix{{0, 0}});

    for(int i = 0; i < size; ++i) {
        double value = aux::gen_double(-1.0, 1.0);
        random_data[i][0][0] = value;
        if(value < 0) {
            labels[i][0][0] = 1;
        }
        else {
            labels[i][0][1] = 1;
        }
    }

    return make_pair(random_data, labels);
}



int main(void) {
    std::cout << "love";
    vector<vector<double>> input{{3,2,1,4,5}};
    vector<vector<double>> actual{{0,0,1,0,0}};
/*
    Matrix love{{1,2,3}};
    auto poop = aux::flatten(love);
    for(auto& item : poop) {
      std::cout << item;
    }
*/
/*
    Dense<> d1(5, 5);
    Relu<> r1;
    Dense<> d2(5, 5);
    Loss<> l;

    auto s1 = d1(input);
    auto s2 = r1(s1);
    auto s3 = d2(s2);
    std::cout << l.comp_loss(s3, actual);
    auto probs = l(s3, actual);
    auto b0 = l.backward_pass(probs, actual);
    auto b1 = d2.backward_pass(b0);
    auto b2 = r1.backward_pass(b1);
    auto b3 = d1.backward_pass(b2);


    // Maybe use a string thing which gets parsed to create the net.
*/


//    Net<double> net;

/*
    Dense<> d1(100,1);
    Relu<> r1;
    Dense<> d2(100, 100);
    Relu<> r2;
    Dense<> d3(2, 100);
    //Dense<> d2(100, 1);

    // make net take unique pointers, or use the string parsing thing.

    Net<double> net({
        &d1,
        &r1,
        &d2,
        &r2,
        &d3
    });
*/

    /*Net<double> net({
        new Dense<>(100, 1),
        new Relu<>(),
        new Dense<>(100, 100),
        new Relu<>(),
        new Dense<>(2, 100)
    });*/

    //auto conv = Conv2d<double>(5, 3, 2, {3,3});
/*
    pair<vector<Matrix>, vector<Matrix>> data = generate_data(1000);

    auto stuff = data.first;
    auto labels = data.second;

    Net<double> net({
        //"conv2d 1 1 1 1 1 1 1"
        "dense 100 1",
        "relu",
        "dense 100 100",
        "relu",
        "dense 2 100"
    });

    for(size_t i = 0; i < stuff.size(); ++i) {
        auto input = stuff[i];
        auto actual = labels[i];
        net.update(input, actual);
    }

    Matrix test {{2}};
    auto val = net.predict(test);
    print(val);
    std::cout << '\n';
    char index = val[0][0] > val[0][1] ? 'n' : 'p';

    std::cout << index;
    std::cout << '\n';

    // SIMPLE CONV TEST. JUST TO GET IT TO RUN.
    Image conv_test {
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1}
        },
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1}
        },
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1}
        },
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1} 
        }
    };
    // number of filters, filter size, stride, input height, wdith, depth, padding 
    Conv2d<double> live(4, 3, 1, 4, 4, 4, 1);
    Conv2d<double> live2(4, 3, 1, 4, 4, 4, 1);
    
    auto out1 = live.forward_pass(conv_test);
    auto out2 = live2.forward_pass(out1);
    auto up1 = live.backward_pass(out2);
    auto up2 = live2.backward_pass(up1);
    for(auto& height : up2) {
        for(auto& width : height) {
            for(auto& depth : width) {
                std::cout << depth << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
*/
    auto mnist_dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>
        (   // mnist data location.
            "/home/afeeney/pet/net/mnist_data/mnist/" 
        );

    Net<double> simp_mnist_net({
        "dense 100 784",
        "relu",
        "dense 100 100",
        "relu",
        "dense 50 100",
        "relu",
        "dense 50 50",
        "relu",
        "dense 50 50",
        "relu",
        "dense 10 50"
    });
    // train the network. 
    unsigned int num_epochs = 6;

    for(unsigned int epoch = 0; epoch < num_epochs; ++epoch) {
        for(int o = 0; o < mnist_dataset.training_labels.size(); ++o) {
            auto image = mnist_dataset.training_images[o];
            auto label = mnist_dataset.training_labels[o];

            vector<double> image_d(image.begin(), image.end());
            vector<double> one_hot_label(10, 0); 
            one_hot_label[label] = 1;
        
            vector<vector<double>> matr_im(1, vector<double>(image_d.size(),
                                                             0));
            for(int d = 0; d < image_d.size(); ++d) {
                matr_im[0][d] = image_d[d];
            }
            vector<vector<double>> matr_lb(1);
            matr_lb[0] = one_hot_label;

            simp_mnist_net.update(matr_im, matr_lb);
        }
    }
    
    int correct = 0;
    
    for(int o = 0; o < mnist_dataset.test_labels.size(); ++o) {
        auto image = mnist_dataset.test_images[o];
        auto label = mnist_dataset.test_labels[o];

        vector<double> image_d(image.begin(), image.end());
        vector<double> one_hot_label(10, 0); 
        one_hot_label[label] = 1;
    
        vector<vector<double>> matr_im(1, vector<double>(image_d.size(),
                                                         0));
        for(int d = 0; d < image_d.size(); ++d) {
            matr_im[0][d] = image_d[d];
        }
        vector<vector<double>> matr_lb(1);
        matr_lb[0] = one_hot_label;


        if(simp_mnist_net.guess_and_check(matr_im, matr_lb)) {
            ++correct;
        }

    }
    double percent_correct = static_cast<double>(correct) / 
                          static_cast<double>(
                                    mnist_dataset.test_labels.size());
    std::cout << "accuracy: " << percent_correct << '\n'; 
    return 0;
}
