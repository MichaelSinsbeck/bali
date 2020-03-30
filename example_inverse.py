"""
Example for setting up and solving an inverse problem using bali

This is the heat-equation problem from our paper, see README.md
"""

import bali as bl
import numpy as np
import matplotlib.pyplot as plt

# load data from disc
content = np.load('example_data/inv_data.npz')
x = content['x']
y = content['y']
variance = content['variance']
data = content['data']

# define model, problem, gpe and sequentialDesign
model = bl.Model(x, y, variance)

problem = bl.Problem(model, data)

gpe = problem.suggest_gpe()

sd = bl.SequentialDesign(problem, gpe)

# do 20 steps in sequential design
n_iter = 20
sd.iterate(n_iter)

# Show error plot
plt.semilogy(sd.error_ll())
plt.xlabel('Number of evaluations')
plt.ylabel('Error (kl-divergence)')
plt.show()

# Show design (=evaluation points)
x1 = np.unique(x[:,0])
x2 = np.unique(x[:,1])
ll_true = problem.compute_loglikelihood()
points = sd.nodes[0].x
plt.pcolor(x1,x2, np.exp(ll_true.reshape(51,51)))
plt.plot(points[:,0], points[:,1], 'w.')
plt.show()