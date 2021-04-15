# MachineLearning
Brennan's machine learning code.

I try hard to use only standard C++ libraries (I hate installing other libraries). Just download and #include the header files for your own program -- should compile with any modern gnu compiler.

Includes:
* Matrix.hpp - Matrix object
* LogisticRegressionModel.hpp - LogisticRegressionModel object
* test.cpp - main program for testing the logistic regression

I kept all members of the class public for direct manipulation by the user. Key parameter members for set-up are:

* learn_rate - the learning rate (default=0.0001)
* cost_threshold - if the *change* in cost drops below this value, breaks (default=0.0000001).
* n_iter - the maximum number of iterations to process (default=1000000).
* print_output - the iteration interval to print a progress message (default=0 for no printed output).

These parameters, sans print_output, can also be set with the class constructor.
```cpp
double learning_rate = 0.001;
double change_cost_threshold = -1;
int num_iterations = 50000000;
int print_epoch_interval = 10000;

LogisticRegressionModel myModel(
  learning_rate,
  change_cost_threshold,
  num_iterations);

myModel.print_output = print_epoch_interval;
```
