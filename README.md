# nn: A toy library for making a neural network in C++.   

Main has a test using the ensemble class for MNIST dataset, it may not be a good test, but just whatever I have been messing around with... Requires -std=c++17. It is a header-only library and the only dependeny is boost/algorithm, so it should be simple to setup. Everything is implemented from scratch. Matrix multiplication to ensemble training.

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
Convolution 
Batch normalization (This runs, but I haven't had a test to run on a cluster with a large enough network.)    
Dropout  
Evolutional Dropout (Again, this runs and accuracy is comparable to dropout but it is difficult to test.)
  
### Activations (treated as layers):  
ReLu  
PReLU  
LeLU  
Sigmoid  
tanh  
softplus   

### Initializers (from torch.nn.init)
xavier normal/uniform
kaiming normal/uniform
  
Most things can be parallelized if I thought it would make sense to do so. So like the dense layer has a parallel version. All activations do. With batch normalization, I'm not really sure how to parallelize it decently or if it really makes sense to for the scale of network I've been using it with.    
  
It is really easy to make an ensemble which trains in parallel and averages all the networks in the ensemble at the end of training or averages them at the end of each epoch. The data_results directory includes some simple graphs of speedup when parallelized. The ensemble also supports mini-batch training!  

Some parts are currently broken. Some parts a bit sloppy. I just play around and add things that seem interesting when I am bored.

## Example ensemble initialization:

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

