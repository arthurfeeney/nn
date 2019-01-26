

# Classifying Sin and Cos Using NN. 

This example is found in classify_sin_cos.cpp.   
It can be compiled with g++ -std=c++17 -O2 classify_sin_cos.cpp -pthread. Note, Boost/algorithm is a dependency.   

Rather than going into detail on what NN does and how it is implemented, this shows a simple use case and points to some things that the library implements. 

## What are we trying to do? 
This is just a little example that I came up with. Essentially, for an ordered pair (a, f(a)), we want to determine what f() is using only the values a and f(a). For this example, we have restricted f() to be sin() or cos(). 

## Generating Train and Test datasets!

In classify_sin_cos.cpp, there is a function

~~~~
std::pair<Dataset, Dataset> generate_data(int size);
~~~~

This function generates a set of data and labels. One Dataset in the returned pair contains data of the form (a, sin(a)) or (a, cos(a)). The other Dataset contains labels. Either (1,0) for sin and (0, 1) for cos). The code is a bit ugly, but that really is all that it does!

~~~~
Dataset sin_data(size / 2);
Dataset cos_data(size / 2);
for(auto& datum : sin_data) 
    datum = rand_sin(g);
for(auto& datum : cos:data)
    datum = rand_cos(g);
~~~~

Just to node, rand_sin() and rand_cos() are custom functions that generate a random pair. Next, we concatenate the sin_data and cos_data datasets. This generates the final dataset. 

~~~~
sin_data.insert(sin_data.end(),
                cos_data.begin(),
                cos_data.end());
Dataset data = sin_data;
~~~~

Now, we need to create labels for this data. It is easy to do since we know that the front half of data is sin and the back half is cos.

~~~~
Dataset labels(size); 
for(int i = 0; i < size / 2; ++i) {
	labels[i] = std::vector {std::vector {1.0, 0.0}};
}
for(int i = size / 2; i < size; ++i) {
	labels[i] = std::vector {std::vector {0.0, 1.0}};
}

~~~~

However, the data is in a bad order, its a lot of sin functions followed by a lot of cos functions. So, we need to shuffle it and the labels together. 

~~~~
std::vector<int> indices(size);
std::iota(indices.begin(), indices.end(), 0);
std::random_shuffle(indices.begin(), indices.end());
for(int i = 0; i < size; ++i) {
	data[i] = data[indices[i]];
	labels[i] = labels[indices[i]];
}
~~~~

The data is now ready to be returned! YES!!!!! But, we are not done yet!

~~~~
return std::make_pair(data, labels);
~~~~

## Learning Something!

Well, using the generate_data function, we want to generate some data!

~~~~
int train_size = 100;
int test_size = 1000;

auto train = generate_data(train_size);
auto test = generate_data(test_size);

Dataset train_data = train.first;
Dataset train_labels = train.second;

Dataset test_data = test.first;
Dataset test_labels = test.second;
~~~~


We can now define a network using the ensemble class that is found in nn/include/ensemble.hpp. We cannot go into everything that the ensemble class contains because there is a LOT that has gone into it... But, we can describe some of the parameters.

~~~~
Ensemble<std::vector<std::vector<double>>, // The type of a SINGLE datum.
		std::vector<std::vector<double>>,
 		Adam<>, Adam optimizer, found in nn/include/Optimizers/adam.hpp
		double>
net(train_data,
	train_labels,
	test_data,
	test_labels,
	1,          // the number of networks in the ensemble
	1e-3,       // the learning rate
	5000,       // steps until learning rate decay
	1,          // learning rate decay (none in this example)
	1,          // batch size
	{
		"dense 20 2",
		"relu",
		"dense 20 20",
		"relu",
		"dense 20 20",
		"relu",
		"dense 2 20"
	});
~~~~

The reason that a single datum in this case is vector<vector<double>> is a bit subtle. The way that the nn/include/Layers.dense.hpp handles processing inputs is always as a matrix. There is no vector type ever. Instead, pretty much the entire library uses a nx1 or 1xn matrix in place of vectors.   

Clearly, the strings inside of the { } defines the networks structure!  

Before we begin training, we need to initialize the networks weights. In this case, we use xavier_normal, which is found in nn/include/Initializers/xavier_normal.hpp.  

~~~~
net.initialize(init::xavier_normal<std::vector<std::vector<double>>, double>, 1);
~~~~

For this example, we restrict the values of a that make up the elements (a, f(a)) to be between 0 and 2pi. This is just to make things a bit easier. Let's try testing it to see what accuracy it can get without any training. By using the function 

~~~~
net.test();
~~~~

We find that with random weights, there is a surprising amount of variance, sometimes as low as 28%, but it is averages to about 50% accuracy. This is expected since the network will basically be output random numbers.  

At long last, we are ready to train the network. We will train it for 10 epochs and set the verbose setting to true. 

~~~~
net.train(10, true, 100);
~~~~

|                    | Accuracy |   
|--------------------|----------|   
| train size=100     | .703     |   
| train size=1,000   | .825     |   
| train size=10,000  | .98      |   

Well, that was easy! Since we restrict it to such a small range, there are no downsides to overfitting, so we can just use any arbitrary size of dataset. Now let's try one more thing. Lets restrict a to be between 0 and 1000pi. This gets much more interesting...   

Lets look at the case where we have 100 training images. It appears that it generally figures out a decent classifier for determining sin or cos; however, sometimes it appears to get it completely backwards. The accuracy is always about 70% correct or about 30% correct. Very intersting. There are NEVER any percent accuracies that are in-between these. This still occurs occasionally when there are 1,000 training images, but not as often. When there are 10,000 training images, this appears to never happen! With 100 training imags, 1,000 training images, and 10,000 training images, the highest percent accuracy they all get is about 70%.   

When the range is between 0 and 1000pi, it must be very hard to "learn" to distinguish between cos and sin. I suppose that is somewhat obvious since the size of numbers will get larger and larger and magnitude has no real effect on the output of sin or cos. but I still think it is a cool result :) I am happy I thought this up.  


