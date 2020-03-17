# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:18:04 2020

@author: sdy
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import product


def sinkhorn_normalized(x, y, epsilon, n, niter):

    Wxy, _, _ = sinkhorn_loss(x, y, epsilon, n, niter)
    Wxx, _, _ = sinkhorn_loss(x, x, epsilon, n, niter)
    Wyy, _, _ = sinkhorn_loss(y, y, epsilon, n, niter)
    return 2 * Wxy - Wxx - Wyy


def sinkhorn_loss(x, y, epsilon, n, niter):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y)  # Wasserstein cost function

    # both marginals are fixed with equal weights
    mu = np.ones((n,), dtype=np.float) / n
    nu = np.ones((n,), dtype=np.float) / n

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations ..................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        r"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + np.expand_dims(u, 1) + np.expand_dims(v, 0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return np.log(np.exp(A).sum(axis=1, keepdims=True) + 1e-8)

    # Actual Sinkhorn loop ...................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    # to check if algorithm terminates because of threshold or max iterations
    # reached
    actual_nits = 0

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (np.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (np.log(nu) - lse(M(u, v).T).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze() ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).T).squeeze() ) + v ) )
        err = np.sum(np.abs(u - u1))

        actual_nits += 1
        if (err < thresh):
            break
    U, V = u, v
    pi = np.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = np.sum(pi * C)  # Sinkhorn cost

    return cost, pi, C


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = np.expand_dims(x, 1)
    y_lin = np.expand_dims(y, 0)
    c = np.sum((np.abs(x_col - y_lin)) ** p, 2)
    return c


def dist_uniform(n_dim, n_cut):
    p = np.linspace(0, 1, n_cut)
    dist = [i for i in product(p, repeat=n_dim)]
    return np.array(dist)


if __name__ == '__main__':
    # Sinkhorn parameters
    epsilon = 0.01
    niter = 100

    """
    n = 200
    N = [n, n]  # Number of points per cloud

    # Dimension of the cloud : 2
    x = np.random.rand(2, N[0]) - .5
    theta = 2 * np.pi * np.random.rand(1, N[1])
    r = .8 + .2 * np.random.rand(1, N[1])
    y = np.vstack((np.cos(theta) * r, np.sin(theta) * r))
    def plotp(x, col): return plt.scatter(
        x[0, :], x[1, :], s=50, edgecolors="k", c=col, linewidths=1)

    # Plot the marginals
    plt.figure(figsize=(6, 6))
    plotp(x, 'b')
    plotp(y, 'r')
    # plt.axis("off")
    plt.xlim(np.min(y[0, :]) - .1, np.max(y[0, :]) + .1)
    plt.ylim(np.min(y[1, :]) - .1, np.max(y[1, :]) + .1)
    plt.title("Input marginals")

    x, y = x.T, y.T
    l1 = sinkhorn_loss(x, y, epsilon, n, niter)
    l2 = sinkhorn_normalized(x, y, epsilon, n, niter)

    print("Sinkhorn loss : ", l1)
    print("Sinkhorn loss (normalized) : ", l2)

    plt.show()

    """
    n_dim = 3
    n_cut = 10
    n1 = 100
    p = np.random.rand(n1, n_dim)
    dp = np.ones((n1,1)) / n1
    p = np.hstack((dp, p)).astype(np.float32)
    q = dist_uniform(n_dim, n_cut)
    dq = np.ones((n_cut**n_dim,1)) / (n_cut**n_dim)
    q = np.hstack((dq, q)).astype(np.float32)
    cost, _, P = cv2.EMD(p, q, cv2.DIST_L2)

