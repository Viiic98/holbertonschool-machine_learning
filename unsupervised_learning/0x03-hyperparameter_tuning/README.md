# Hyperparameter Tuning

### Useful resources
- [Tasks 0 and 1](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes.ipynb?flush_cache=true)
- [Bayesian](./http://krasserm.github.io/2018/03/21/bayesian-optimization/)
- [Bayesian EI](./https://thuijskens.github.io/2016/12/29/bayesian-optimisation/)


### Install GPy and GPyOpt
```pip install --user GPy```

```pip install --user gpyopt```

### Tasks

#### [Initialize Gaussian Process](./0-gp.py)
- Create the class GaussianProcess that represents a noiseless 1D Gaussian process

#### [Gaussian Process Prediction](./1-gp.py)
- Based on 0-gp.py, update the class GaussianProcess
#### [Update Gaussian Process](./2-gp.py)
- Based on 1-gp.py, update the class GaussianProcess

#### [Initialize Bayesian Optimization](./3-bayes_opt.py)
- Create the class BayesianOptimization that performs Bayesian optimization on a noiseless 1D Gaussian process

#### [Bayesian Optimization - Acquisition](./4-bayes_opt.py)
- Based on 3-bayes_opt.py, update the class BayesianOptimization

#### [Bayesian Optimization](./5-bayes_opt.py)
- Based on 4-bayes_opt.py, update the class BayesianOptimization
