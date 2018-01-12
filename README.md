# nn
A toy library for making a neural network in C++.   

Main has a test using the ensemble class for MNIST datase, it may not be a good test, but just whatever I have been messing around with... Needs g++-7 -std=c++17 or clang 3.9 or some other equivalent compiler.   

Implements:   
Optimizers:    
Adam   
Adagrad    
SGD  
Momentum  
Nesterov Momentum  
RMSProp  
   
Layers:  
dense    
Convolution (I don't think this is working...)  
Batch normalization (I'm pretty sure this is working. I haven't had an opportunity to test it with a large enough network yet.)    
Dropout (this is basically a layer)  
  
Activations (treated as layers):  
ReLu  
PReLU  
LeLU  
Sigmoid  
tanh  
softplus   
  
Most things can be parallelized if I thought it would make sense to do so. So like the dense layer has a parallel version. All activations do. Convolution doesn't because it doesn't really work. With batch normalization, I'm not really sure how to parallelize it decently or if it really makes sense to for the scale of network I've been using it with...    
  
It is really easy to make an ensemble which trains in parallel and averages all the networks in the ensemble at the end of training or averages them at the end of each epoch. Further, each network in the ensemble can be parallelized. The data_results directory includes some simple graphs of speedup when parallelized.   
  
The ensemble also supports mini-batch training!  
I hope I'm not forgetting to mention anything! Although I'd be a little suprised if anyone ever read this far down the readme (lol)! :o  
