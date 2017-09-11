#include <iostream>
#include <vector>
#include <memory>
#include <utility>

#include "aux.hpp"
#include "layers.hpp"
#include "net.hpp"

using std::vector;
using std::make_unique;
using std::pair;
using std::make_pair;

void print(vector<vector<double>>& m) {
    for(auto& row : m) {
        for(auto& item : row) {
            std::cout << item << ' ';
        }
        std::cout << '\n';
    }
}

using Matrix = vector<vector<double>>;
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
  
    vector<vector<double>> input{{3,2,1,4,5}};
    vector<vector<double>> actual{{0,0,1,0,0}};
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

    pair<vector<Matrix>, vector<Matrix>> data = generate_data(10000);

    auto stuff = data.first;
    auto labels = data.second;

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

    Net<double> net({
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

    // NEED TO USE POINTERS FOR THIS STUFF :(
    /*vector<Layer<>*> stuff {
        &d1,
        &r1
    };*/
    //auto poo = (*net[0])(input);
   
/*    
    auto poo = net.predict(input);
    print(poo);

    Net<> net;
    net.predict(input);
*/  
  //net[0](input);

    //print(net[0]);
    /*
    Layer<> poo = net[0];
    auto val = poo(input);
    print(val);
    */
    return 0;
}
