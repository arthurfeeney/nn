# nn: A toy library for making neural networks in C++.   

Main has a test using the ensemble class for MNIST dataset, it may not be a good test, but just whatever I have been messing around with... Requires -std=c++17. It is a header-only library and the only dependeny is boost/algorithm, so it should be simple to setup. Everything is implemented from scratch; matrix multiplication up :)

## In-Depth Examples:

An in-depth example can be found in the examples/ directory. There is a file called classify_sin_cos.cpp and that directories README contains a walkthrough of what the source code contains.

## Implements:   
### Optimizers:    
Adam   
Adagrad    
SGD  
Momentum  
Nesterov Momentum  
RMSProp  
   
### Layers:  
dense    
Convolution (Currently Broken, tried to implement it in a different way...)     
Batch normalization (This runs, but I haven't had a test to run on a cluster with a large enough network.)    
Dropout  
Evolutional Dropout
  
### Activations (treated as layers):  
ReLu  
PReLU  
LeLU  
Sigmoid  
tanh  
softplus   

### Initializers (same as torch.nn.init)
xavier normal  
xavier uniform  
kaiming normal  
kaiming uniform  
  
Most things can be parallelized if I thought it would make sense to do so. For instance, the dense layer has a parallel version. All activations do. With batch normalization, I'm not really sure how to parallelize it decently or if it really makes sense to for the scale of network I've been using it with.    
  
It is really easy to make an ensemble that trains in parallel and averages all the networks in the ensemble at the end of training or averages them at the end of each epoch. The data_results directory includes some simple graphs of speedup when parallelized. The ensemble also supports mini-batch training!  

Some parts are currently broken. Some parts a bit sloppy. I just play around and add things that seem interesting when I am bored.


## Example Ensemble Construction:

~~~~
Ensemble<vector<vector<double>>, // type of input data  
         vector<vector<double>>, // type of label data
         Adam<>,  // network optimizer, in include/Optimzer/adam.hpp.  
         double>  // type of networks weights. 
ensemble  
(  
    train_data,  
    train_labels,  
    test_data,   
    test_labels,  
    ensemble_size,  
    1e-3, // learning rate  
    64,   // batch size  
    {  
        "dense 1024 784",  // standard fully-connected network layer   
        "relu",            // activation function  
        "bn 1e-5, 0.1",    // batch normalization (I think this works...)   
        "dense 1024 1024",  
        "relu",  
        "bn 1e-5, 0.1",  
        "dense 1024 1024",  
        "relu",  
        "dropout 0.5",  
        "dense 10 1024"  
    }  
    number_threads // the number of threads EACH network in the ensemble can use.  
);  
~~~~

## Examples of Tasks that are handled entirely by the Ensemble class:
### Initialization with Xavier Normal:  
~~~~
ensemble.initialize(init::xavier_normal<std::vector<std::vector<double>>, double>, 1);
~~~~

### Training:  
Train the ensemble for 2 epochs, with the verbose setting true so that it prints every 1000th input.
~~~~
ensemble.train(2, true, 1000);
~~~~

### Ensemble Accurary on Test Dataset:  
~~~~
double test_acc = ensemble.test();
~~~~

## MNIST Example:
This is just to show that the implementation is able to work on more rigorous datasets. The network below is able to achieve decent accuracy on a classic dataset. The settings are not at all optimal, but it is still about to achieve about 97% accuracy on the MNIST test dataset. 

~~~~
Ensemble<vector<vector<double>>, 
         vector<vector<double>>, 
         Adam<>,
         double> 
net 
(
    //data_to_im(mnist_dataset.training_images, 28, 28),
    get_all_data(mnist_dataset.training_images),
    get_all_label(mnist_dataset.training_labels),
    //data_to_im(mnist_dataset.test_images, 28, 28),
    get_all_data(mnist_dataset.test_images),
    get_all_label(mnist_dataset.test_labels),
    1, // ensemble size
    1e-4, // learning rate
    1, // num steps between learning rate decay
    1, // amount to scale learning rate (No decay in this example). 
    32, // batch size
    {
        "dense 676 784",
        "relu",
        "dense 300 676",
        "relu",
        "dense 300 300",
        "relu",
        "dense 300 300",
        "relu",
        "dense 10 300"
    },
    1 // the 1 network in the ensemble uses 1 thread
);

net.initialize(init::xavier_normal<std::vector<std::vector<double>>, double>, 1);

net.train(10, true, 1000);

double test_accuracy = net.test();
~~~~



