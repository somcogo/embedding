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

def draw_on_axes(ax: Axes, vectors: torch.Tensor, var: torch.Tensor = None, mu: torch.Tensor = None, comps: int = 2, bins=100, cmap1=matplotlib.cm.hot, cmap2=matplotlib.cm.YlGn):
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.hist2d([vector[1] for vector in vectors], [vector[0] for vector in vectors], bins=bins, cmap=cmap1)
    if mu is not None:
        x = np.linspace(-5, 5, bins)
        y = np.linspace(-5, 5, bins)
        z = torch.zeros(bins, bins)
        for comp_id in range(comps):
            z += calc_multivar_normal_distr(x, y, mu[comp_id], var[comp_id])
        xv, yv = np.meshgrid(x, y)
        ax.contour(xv, yv, z, levels=4, cmap=matplotlib.cm.YlGn, alpha=0.6)
    return ax