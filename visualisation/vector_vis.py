import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch

def calc_multivar_normal_distr(X, Y, mu, sigma):
    mu = mu.float()
    sigma = sigma.float()
    sigma_inv = torch.linalg.inv(sigma)
    z = torch.zeros((len(X), len(Y)))
    
    for x_ndx, x in enumerate(X):
        for y_ndx, y in enumerate(Y):
            vector = torch.tensor([x, y], dtype=torch.float)
            sigma_inv_x_minus_mu = torch.matmul(sigma_inv, vector-mu)
            exponent = -torch.matmul((vector-mu), sigma_inv_x_minus_mu)
            z[x_ndx, y_ndx] = exponent
    z = torch.exp(z)
    return z / z.sum()

def draw_on_axes(ax: Axes, vectors: torch.Tensor, var: torch.Tensor = None, mu: torch.Tensor = None, comps: int = 2, bins=100, cmap1=matplotlib.cm.hot, cmap2=matplotlib.cm.YlGn, mode='hist', c=None, s=20):
    if mode == 'hist':
        ax.hist2d(vectors[:,1], vectors[:,0], bins=bins, cmap=cmap1)
    elif mode == 'scatter':
        ax.set_xlim(bins[0][0], bins[0][-1])
        ax.set_ylim(bins[1][0], bins[1][-1])
        cmap = matplotlib.cm.plasma
        ax.scatter(vectors[:,1], vectors[:,0], edgecolors='none', alpha=0.5, c=c, s=s, label=c)
    if mu is not None:
        x = bins[0]
        y = bins[1]
        z = torch.zeros(len(bins[0]), len(bins[1]))
        for comp_id in range(mu.shape[0]):
            z += calc_multivar_normal_distr(x, y, mu[comp_id], var[comp_id])
        xv, yv = np.meshgrid(x, y)
        ax.contour(xv, yv, z, levels=4, cmap=matplotlib.cm.YlGn, alpha=0.6)
    return ax