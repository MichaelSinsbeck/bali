"""
Example for setting up and solving a Bayesian model selection problem using bali

This is the curve-fitting problem given by Schöniger et al:
    
A. Schöniger, T. Wöhling, L. Samaniego, and W. Nowak, Model selection on 
solid ground: Rigorous comparison of nine ways to evaluate Bayesian model 
evidence, Water Resources Research, 50 (2014),pp. 9484–9513
"""

import bali as bl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# load data from disc and define model functions
content = np.load('example_data/selection_data.npz')
x_linear = content['x_linear']
x_cosine = content['x_cosine']
data = content['data']
variance = content['variance']
grid = content['grid']

# Linear model y = a*x+b
def f_linear(p):
    return p[0] * grid + p[1]

# Nonlinear model y = a*cos(b*x+c)+d
def f_cosine(p):
    return p[0] * np.cos(p[1] * grid + p[2]) + p[3]

# define models, problem, gpes and sequentialDesign
model_linear = bl.Model(x_linear, f_linear, variance)
model_cosine = bl.Model(x_cosine, f_cosine, variance)

problem = bl.Problem([model_linear, model_cosine], data)

gpes = problem.suggest_gpes()

sd = bl.SequentialDesign(problem, gpes)

# do a total of 50 steps in sequential design
# we can switch between different model-time allocation strategies
sd.allocation = 'alternation'
sd.iterate(20)

sd.allocation = 'acq'
sd.iterate(30)

# Show error plot
plt.semilogy(sd.error_lbme())
plt.xlabel('Number of evaluations')
plt.ylabel('Error (kl-divergence)')
plt.show()

# Show sequence of BME-estimates
plt.plot(np.exp(sd.lbme))
plt.xlabel('Number of evaluations')
plt.ylabel('BME estimates')
plt.show()