# bali - A python package for Bayesian likelihood estimation

If you use this package for your research, please cite the following article:

[Sinsbeck, Nowak: Sequential Design of Computer Experiments for the Solution of Bayesian Inverse Problems (2017) SIAM JUQ.](https://doi.org/10.1137/15M1047659)

If you have any questions or feedback on this package, please contact one of the authors, [here](https://www.iws.uni-stuttgart.de/institut/team/Sinsbeck-00001/) or [here](
https://www.iws.uni-stuttgart.de/institut/team/Nowak-00008/)

This package uses `numpy` and `scipy`. For some rather uncommon ways of handling hyperparameters (namely `average` and `linearize`, see below), also the package `emcee` (2.2.1) is used. If you don't know whether you need `emcee` or not, you probably do not need it.

## What this package is

Consider a model of the form f(x) + e, where f is a model function (for example a simulator), x represents unknown input parameters and e is an additive normally distributed measurement error.

After measuring some output data d, we would like to compute the likelihood of x given d: L(x|d)

This likelihood is large, if f(x) is close to d.

This python package helps compute the likelihood function L(x|d) using a sequential design of experiments and Gaussian processes. It is specifically designed to provide good estimates of L(x|d) with as few evaluations of f as possible.

This package works best, when f is a computationally expensive function, such as a simulator, and if the number of parameters of f is small or moderate (up to 10 or so).

Solving optimization with Gaussian processes is called "Bayesian Optimization". In the same spirit, I called this method "Bayesian likelihood estimation".

The likelihood is useful in two types of problems: (1) Bayesian inverse problems and (2) Bayesian selection problems. This package provides the functionality to solve both of these types.

## Quickstart
This repository contains examples to show how the package is used. Study the code inside to see, how this package is used.

`example_inverse.py` shows how to set up an inverse problem and solve it using bali

`example_selection.py` shows how to set up a Bayesian model selection problem and solve it.

## How to use

To use bali, use the following steps:

1. create a model (or multiple models)
2. create a problem (either inverse problem oder selection problem)
3. create a gpe (a Gaussian process emulator) for each model
4. create a `SequentialDesign` object.
5. run `SequentialDesign.iterate`

These steps are explained in the following sections.

### Create a Model

In bali a "model" refers to a stochastic model. It consists of uncertain input parameters X, a model function f and an additive normally distributed error e: Z = f(X)+e.

After importing bali, create a model like that:
```python
import bali as bl
model = bl.model(x, f, variance)
```
Input parameters are:

- `x` is a sample of X. It must be a numpy-array of size `(n_x, n_input)`. If the model is one-dimensional (`n_input == 1`), then `x` can also be a vector of size `(n_x)`.
- `f` is the model function. It can either be a function that takes an input of size `(n_input)` and outputs an array of size `(n_output)`, or it can be a table of values of size `(n_x, n_output)`, containing the model output for the input vectors in `x`. The latter makes only sense in numerical studies, because knowing the output for all inputs would make the use of a sequential design useless.
- and `variance` denotes the variance of e. It can either be a scalar or a vector of size `(n_output)`.

### Create a Problem

The bali class `Model` can both represent a Bayesian inverse problem or a Bayesian model selection problem.

#### Inverse Problems

To create an inverse problem, we have to pass it a model and a data vector.
```python
problem = bl.Problem(model, data)
```
- `data` is a vector of size `(n_output)`.

We can compute the solution of this inverse problem using the functions
```python
likelihood = problem.compute_likelihood()
loglikelihood = problem.compute_loglikelihood()
```
Both functions return a vector of size `(n_x)`. Each entry contains the (log)likelihood of the corresponding input vector in `x`.

#### Model Selection Problems

And to create a model selection problem, we have to pass a list of models and a data vector
```python
problem = bl.Problem([model1, model2], data)
```
We can compute the solution of this selection problem using the function
```python
lbme = problem.compute_lbme()
```
This will return a vector of size `(n_models)` containing the log-BMEs of the models.

### Create a GPE

Next, we need to specify the GPE (or GPEs) we want to use to emulate the model (or models). There are a couple of options:

#### Select GPE by hand

bali currently only offers GPE with mean zero. There are two possible covariance function to choose from: squared-exponential and Matèrn. GPE objects can be created as follows:
```python
gpe1 = bl.GpeMatern(l, sigma_squared, nu, n_output, anisotropy = None)
gpe2 = bl.GpeSquaredExponential(l, sigma_squared, n_output, anisotropy = None)
```
The first arguments are GPE-hyperparameters:

- `l` is correlation length. It can be a scalar (same correlation length for all dimensions) or a vector of size `(n_input)`
- `sigma_squared` is the variance. A scalar
- `nu` (in the Matèrn case) is a smoothness parameter. A scalar.

If hyperparameters are to be identified automatically, then we can enter "soft bounds" for the hyperparameters. In this case, we replace each entry by a list of two values `[lower_bound, upper_bound]`. This will generate a log-normal distribution for the corresponding hyperparameter and will set the two bounds (roughtly) as the 2.5-percentiles. In that sense, these bounds are "soft".

For example, if our model has three input parameters and five output parameters, we could create a gpe with soft bounds as follows:
```python
l = [[0.1, 10], [1, 100], [0.1, 1]]
sigma_squared = [0.001, 100]
nu = [0.5, 10]
gpe = bl.GpeMatern(l, sigma_squared, nu, 5)
```

Setting lower and upper bounds to the same value will lead to fixed hyperparameters. It is possible to mix fixed and free hyperparameters.

The other two arguments are:

- `n_output` - the dimension of the models output.
- `anisotropy` - Controls the anisotropy. If the model input is higher dimensional, but only one value for `l` is given, by default, the GPE assumes the same correlation length for all dimensions. If separate correlation length parameters for each dimension are expected, set anisotropy to the dimension of the input.

For a model with three input parameters and 5 output parameters, we can write
```python
gpe = bl.GpeMatern([0.1, 10], 2, 0.5, 5, anisotropy = 3)
```
#### Let bali suggest a GPE

Once a problem is defined, we can use the information from the problem to obtain a "suggested" GPE. It uses the input `x` to suggest some bounds for the correlation length parameters and uses the measurement data to suggest bounds for `sigma_squared`.

For inverse problems, we can use

```python
gpe1 = problem.suggest_gpe(gpe_type = 'squared_exponential')
gpe2 = problem.suggest_gpe(gpe_type = 'matern')
```
And for selection problems, we need a gpe for each model, so we use (note, the only difference is the plural s in "gpes")
```python
gpe1 = problem.suggest_gpes(gpe_type = 'squared_exponential')
gpe2 = problem.suggest_gpes(gpe_type = 'matern')
```
This will create a list of gpes.

#### GPE Options
If hyperparameters are not fixed, then by default they will be identified via maximum-a-posteriori (MAP) estimate. Two other ways of handling hyperparameters are available: `linearize` and `average`. In both cases, hyperparameters will be drawn in form of a sample from their posterior distribution (via MCMC). In the case `linearize` the resulting (non-Gaussian) random field will be linearized, such that it is Gaussian again. We obtain another GPE. In the case `average`, the sample of hyperparameters will be kept. When using in a sequential design, the acquisition function will then be averaged.

Select one of the three modes as follows:
```python
gpe.hyperparameter_handling = 'map'
gpe.hyperparameter_handling = 'average'
gpe.hyperparameter_handling = 'linearize'
```

The defaul sample size of the MCMC is 10. It can be changed using
```
gpe.n_walkers = 10
```

### Create a SequentialDesign-Object

With the problem and gpe in place, we now create a sequential-design-object

```python
sd = bl.SequentialDesign(problem, gpe, acq='inverse', allocation='acq')
```

- `problem` is a problem object, see above
- `gpe` is a gpe-object, see above, or - in case of a selection problem - a list of gpe-objects
- `acq` defines the acquisition function. Possible values are `inverse`, `random` and `min_variance`. `inverse` will use the adaptive sampling strategy presented in my paper (see top of the readme). `random` will sample randomly and `min_variance` will minimize the gpes total variance.
- `allocation` controls the model-time allocation strategy, only relevant for model-selection problems. Possible values are `acq`, `random`, `alternate`. `acq` picks the model with the highest maximum value in the acquisition function. `random` picks the model randomly and `alternate` alternates between the models evenly (e.g. 0,1,2,0,1,2,0,1,2,...).

The acquisition function can be changed at any time using `sd.set_acquisition_function(acq)` with the same possible values as above (`inverse`, `random`, `min_variance`).

The model-time allocation strategy can be switched by `sd.allocation = allocation` with the same possible values as above (`acq`, `random`, `alternate`).

The sequential design computes integrals and optimizes only on a subsample. The size of this subsample is, by default, 500. It can be changed by
```python
sd.n_subsample = 500
```
Set it to `np.inf` to disable subsampling.

### Start Iteration

Finally, we can start the sequential-design procedure. We run
```python
sd.iterate(n)
```
This will run `n` iterations of the sequential design. Each iteration consists of: 

1. finding the maximum of the acquisition function (for each model) 
2. selecting a model to evaluate (using the model-time allocation strategy) 3. selecting a point in the models input domain (using the acquisition function maximum)
4. evaluating this model at this point
5. conditioning the gpe to the new evaluation
6. making an estimate of the target quantity (likelihood or BME)

When the iterations have finished, we can extract results as follows:

- `sd.ll` returns a list that contains the loglikelihood for each model and for each iteration. `sd.ll[k]` is the list for model `k`. It is an array of size `(n_iter, n_x)`. `n_iter` is the total number of iterations done.
- `sd.lbme` returns an array containing the log-BME values for all iterations. It is an array of size `(n_iter, n_models)`.
- `sd.nodes` is a list of node-objects. These represent the evaluation points for each of the model. `sd.nodes[k]` holds the evaluations for model `k`. The design points are in `sd.nodes[k].x` and the model responses are stored in `sd.nodes[k].y`.
- `sd.k_list` is a list of model indices (which model was evaluated in which iteration). This only makes sense for model-selection problems.

If you need the loglikelihood as a function of x (instead of as a vector), then use
```python
ll_function = sd.loglikelihood_function()
whatever = ll_function(x)
```
In the case of multiple models, also pass the model index `sd.loglikelihood_function(k)`

This will always return the loglikelihood estimate after the last iteration. Intermediate likelihood estimate can not be recovered.

If the model functions themselves are fast, such that we can tabulate the whole model functions, then we can compute reference solutions and therefore also errors. Caution: only do this, if the models are fast.

- `sd.error_ll()` computes the errors in the (log)likelihood [for inverse problems] in terms of kl-divergence. The output is of size `(n_iter)`. If there are multiple models (in a selection problem) we have to pass the model index `sd.error_ll(k)`
- `sd.error_lbme()` computes the errors in the (log)-BMEs in terms of kl-divergence. The output is of size `(n_iter)`.


## Array sizes
It is important that all of the parameters have the correct size. We need the following numbers:

- `n_x` - This is the number of points representing the input `x` of a model.
- `n_input` - The dimension of the input space (=number of parameters of a model)
- `n_output` - The dimension of the output of a model. Also the size of the measurement data
- `n_nodes` - Number of model evaluations done for one model.
- `n_models` - Number of models in a problem
- `n_iter` - Number of iterations


