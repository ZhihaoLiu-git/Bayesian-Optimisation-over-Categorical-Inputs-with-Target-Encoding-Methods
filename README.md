# Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods
Many real-world functions are defined over both categorical and continuous variables. To optimize such functions, we propose a new method that encodes the categorical variables as a continuous variables, where in each category corresponds to function value.

# Unbalanced distribution of data set from BO 

# Benchmark
(definition: https://github.com/WholeG/Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods/blob/8761569e255a3da6d19ff95ce35b4500bb2fb2ed/syntheticFunctions.py)
#### MLP(multi-layer perceptron)
- optimize mse of MLP regression model

1. categorical variables:
  * activation function: {0: 'logistic', 1: 'tanh', 2: 'relu'}
	* learning rate: {0: 'constant', 1: 'invscaling', 2: 'adaptive'}
	* optimization solver: {0: 'sgd', 1: 'adam'}
  * early_stopping: {0: True, 1: False}
2. continuous variables:
	* hidden_layer_sizes: (1, 200)
	* alpha: (0.0001, 1)
	* tol: (0.00001, 1)

![avatar](https://github.com/WholeG/Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods/blob/main/pics/MLP_performance.jpg)

#### hartmann6
1. categorical variables:
  * converts a continuous variables to 17 choices
2. continuous variables:
  * 2 continuous variable 
![avatar](https://github.com/WholeG/Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods/blob/main/pics/HM_performance.jpg)
