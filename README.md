# Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods
Many real-world functions are defined over both categorical and continuous variables. To optimize such functions, we propose a new method that encodes the categorical variables as a continuous variables, where in each category corresponds to function value.


## Benchmark



- with $d$ = 2, $c$ = [3, 3, 2, 2]. Similar \emph{SVM-1C-Boston}, returns logarithmic mean square error as $y$. 
- the green points are train set
- the gray plane is the true function branin, use 400 points by meshgrid()
- the red plane is the prediction of GP on design_domain(all data train set)
- the red vertical line and the blue star point tell us the next point that should be added
![avatar](https://github.com/WholeG/Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods/blob/main/pics/MLP_performance.jpg)
![avatar](https://github.com/WholeG/Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods/blob/main/pics/mlp_hartmann.pdf)
