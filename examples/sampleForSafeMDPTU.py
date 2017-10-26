from __future__ import division, print_function, absolute_import

import time
import networkx as nx

import matplotlib
matplotlib.use('TKAgg')

import GPy
import matplotlib.pyplot as plt
import numpy as np
import examples.sampleConstants as constants
from collections import defaultdict

from src.grid_worldTU import (compute_true_safe_set, compute_S_hat0, compute_true_S_hat, draw_gp_sample, GridWorldTU)

# Define GP
noise = 0.001
kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1.,ARD=True)
lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
lik.constrain_bounded(1e-6, 10000.)

# Sample and plot world
altitudes, coord = draw_gp_sample(kernel, constants.world_shape, constants.step_size)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(coord[:, 0], coord[:, 1], altitudes)
plt.show(block=False)

# Define coordinates
n, m = constants.world_shape
step1, step2 = constants.step_size
xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),np.linspace(0, (m - 1) * step2, m),indexing="ij")
coord = np.vstack((xx.flatten(), yy.flatten())).T

# Data to initialize GP
n_samples = 1
ind = np.random.choice(range(altitudes.size), n_samples)
X = coord[ind, :]
Y = altitudes[ind].reshape(n_samples, 1) + np.random.randn(n_samples,1)
gp = GPy.core.GP(X, Y, kernel, lik)

# Initialize safe sets
S0 = np.zeros((np.prod(constants.world_shape), 5), dtype=bool)
S0[:, 0] = True
S_hat0 = compute_S_hat0(np.nan, constants.world_shape, 4, altitudes, constants.step_size, constants.h)

# Define SafeMDPTU object
x = GridWorldTU(gp, constants.world_shape, constants.step_size, constants.beta, altitudes, constants.h, S0, S_hat0, constants.L)

# Insert samples from (s, a) in S_hat0
tmp = np.arange(x.coord.shape[0])
s_vec_ind = np.random.choice(tmp[np.any(x.S_hat[:, 1:], axis=1)])
tmp = np.arange(1, x.S.shape[1])
actions = tmp[x.S_hat[s_vec_ind, 1:].squeeze()]
for i in range(3):
    x.add_observation(s_vec_ind, np.random.choice(actions))

# Remove samples used for GP initialization
x.gp.set_XY(x.gp.X[n_samples:, :], x.gp.Y[n_samples:])

t = time.time()
x.plot_S_iteration(x.S_hat)
for i in range(constants.num_iterations):
    x.update_sets()
    next_sample = x.target_sample()
    x.add_observation(*next_sample)
    x.plot_S_iteration(x.S_hat, 0, 'Safe set for action {0}, iteration: ' + str(i))
    print("Iteration: " + str(i))

#plotting the graph
plt.figure(3)
plt.clf()
pos = nx.spring_layout(x.graph,k=10)

edge_labels_action = nx.get_edge_attributes(x.graph,'action')
#print(edge_labels_action)
edge_labels_safe = nx.get_edge_attributes(x.graph,'safe')
for key in edge_labels_safe:
    edge_labels_safe[key] = edge_labels_safe[key].tolist()
#print(edge_labels_safe)

#combining dictionaries to make a single edge label - Ex. [action, safe, probability]
edge_labels = defaultdict(list)
for d in (edge_labels_action, edge_labels_safe):  # you can list as many input dicts as you want here
    for key, value in d.items():
        edge_labels[key].append(value)

print(edge_labels)

nx.draw_networkx(x.graph, pos)
nx.draw_networkx_edge_labels(x.graph, pos, edge_labels, font_size=7)

plt.show(block=False)



print(str(time.time() - t) + "seconds elapsed")

true_S = compute_true_safe_set(x.world_shape, x.altitudes, x.h)
true_S_hat = compute_true_S_hat(x.graph, true_S, x.initial_nodes)

# Plot safe sets
x.plot_S(x.S_hat,0,4,'S_hat:   ')
x.plot_S(true_S_hat,0,5,'true_S_hat:   ')

# Classification performance
print(np.sum(np.logical_and(true_S_hat, np.logical_not(x.S_hat))))  # in true S_hat and not S_hat
print(np.sum(np.logical_and(x.S_hat, np.logical_not(true_S_hat))))
#input('type q to exit')
