#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include <chrono>
#include <type_traits>

#include "include/aux.hpp"
#include "include/layers.hpp"
#include "include/net.hpp"
#include "include/ensemble.hpp"
#include "mnist_data/mnist/include/mnist/mnist_reader.hpp"
#include "include/conv2d.hpp"

using std::vector;
using std::make_unique;
using std::pair;
using std::make_pair;

using Matrix = vector<vector<double>>;
using Image = vector<Matrix>;

void print(Matrix& m) {
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

template<typename T>
vector<vector<double>> conv_mnist_data(const T& datum) {
    auto image = datum;
    vector<double> image_d(image.begin(), image.end());
    vector<vector<double>> matr_im(1, vector<double>(image_d.size(), 0));
    for(size_t d = 0; d < image_d.size(); ++d) {
        matr_im[0][d] = image_d[d];
    }
    return matr_im;
}

template<typename T>
vector<vector<double>> conv_mnist_label(const T& l) {
    auto label = l;
    vector<double> one_hot_label(10, 0);
    one_hot_label[label] = 1;
    vector<vector<double>> matr_lb(1);
    matr_lb[0] = one_hot_label;
    return matr_lb;
}

template<typename T>
vector<vector<vector<double>>> get_all_data(const T& data) {
    vector<vector<vector<double>>> ret(data.size());
    for(size_t d = 0; d < data.size(); ++d) {
        ret[d] = conv_mnist_data(data[d]);
    }
    return ret;
}

template<typename T>
vector<vector<vector<double>>> get_all_label(const T& labels) {
    vector<vector<vector<double>>> ret(labels.size());
    for(size_t d = 0; d < labels.size(); ++d) {
        ret[d] = conv_mnist_label(labels[d]);
    }
    return ret;
}

template<typename T>
auto flat_to_im(const T& flat, size_t height, size_t width) {
    vector<vector<vector<double>>> im(width, vector<vector<double>>(height,
                                       vector<double>(1, 0)));
    size_t index = 0;
    for(auto& height : im) {
        for(auto& width : height) {
            for(auto& val : width) {
                val = flat[index];
                ++index;
            }
        }
    }
    return im;
}

template<typename T>
auto data_to_im(const T& data, size_t height, size_t width) {
    vector<vector<vector<vector<double>>>> images(data.size());

    for(size_t i = 0; i < data.size(); ++i) {
        images[i] = flat_to_im(data[i], height, width);
    }
    return images;
}


int main(int argc, char** argv) {
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
            "/home/afeeney/pet/net/mnist_data/mnist/");

/*
    Net<double> simp_mnist_net(
    1e-3,        
    {
        "dense 100 784",
        "relu",
        "dense 100 100",
        "relu",
        "dense 100 100",
        "relu",
        "dense 50 100",
        "relu",
        "dense 50 50",
        "relu",
        "dropout .5", // dropout should hurt accuracy a bit since network is so small.
        "dense 10 50"
    });
    // train the network. 
    unsigned int num_epochs = 1;

    for(unsigned int epoch = 0; epoch < num_epochs; ++epoch) {
        for(size_t o = 0; o < mnist_dataset.training_labels.size(); ++o) {
            if(o % 10000 == 0) {
                std::cout << "epoch: " << epoch << " step: " << o << '\n';
            }
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
    
        vector<vector<double>> matr_im(1, vector<double>(image_d.size(), 0));
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
  */

    //decltype(mnist_dataset.training_images)
    //decltype(mnist_dataset.training_labels), 

    /*
    Ensemble<vector<vector<double>>,
             vector<vector<double>>,
             double> love(
            get_all_data(mnist_dataset.training_images),
            get_all_label(mnist_dataset.training_labels),
            get_all_data(mnist_dataset.test_images),
            get_all_label(mnist_dataset.test_labels),
            4, // ensemble size 
            1e-3, // learning rate
            128, // batch size.
            {
                "dense 300 784",
                "relu",
                //"dense 300 784",
                //"relu",
                "dense 10 300"
*/
                /*"relu",
                "dense 100 100",
                "relu",
                "dense 50 100",
                "relu",
                //"dropout .5", 
                "dense 10 50"*/
 /*           }); // network
    
    auto start = std::chrono::system_clock::now();
    love.train(1, true, 10000);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "time: " << elapsed_seconds.count() << '\n';
    std::cout << love.test() << '\n';
*/
    //Conv2d<double> live(4, 3, 1, 28, 28, 1, 1, 1e-3);
    //live.forward_pass(data_to_im(mnist_dataset.training_images, 28, 28)[0]);
    

    Ensemble<vector<vector<double>>,
             vector<vector<double>>,
             double> conv_net 
    (
        //data_to_im(mnist_dataset.training_images, 28, 28),
        get_all_data(mnist_dataset.training_images),
        get_all_label(mnist_dataset.training_labels),
        //data_to_im(mnist_dataset.test_images, 28, 28),
        get_all_data(mnist_dataset.test_images),
        get_all_label(mnist_dataset.test_labels),
        2, // ensemble size
        1e-4, // learning rate
        8, // batch size
        {
            //"conv2d 2 3 1 28 28 1 0",
            //"dense 50 1352",
            "dense 100 784",
            //"leaky .00001",
            "relu",
            "dense 50 100",
            //"leaky .00001",
            "relu",
            //"dropout .5",
            "dense 10 50"
        },
        1// number of threads per network in ensemble.
    );
    auto start = std::chrono::system_clock::now();
    conv_net.train(2, true, 1000);
    std::cout << conv_net.test();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << '\n' << "time: " << elapsed_seconds.count() << '\n';
    
    /*
    Conv2d<double> live(4, 3, 1, 4, 4, 4, 0, 1e-4);
    
    vector<vector<vector<double>>> i {
        {
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1}
        },
        {
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1}
        },
        {
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1}
        },
        {
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1},
            {1,1,1,1}
        }
    };
    
    auto ni = live.forward_pass(i);

    for(auto& c : ni) {
        for(auto& r : c) {
            for(auto& val : r) {
                std::cout << val << ' ';
            }   
            std::cout << '\n';
        }
        std::cout << '\n';
    }
*/

    return 0;
}
