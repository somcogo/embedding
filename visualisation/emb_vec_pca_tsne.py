import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def vis_embedding(trn_path, ft_path, trn_sites, ft_sites, deg, x_dim=0, y_dim=1, legend=True, title_str=''):
    trn_dict = torch.load(trn_path, map_location='cpu')
    embs_trn = np.zeros((trn_sites, trn_dict[0]['site_model_state']['embedding'].shape[0]))
    for i in range(trn_sites):
        embs_trn[i] = trn_dict[i]['site_model_state']['embedding']
    

    tsne_emb_trn = TSNE(perplexity=min(5, trn_sites)).fit_transform(embs_trn)
    pca_trn = PCA(n_components=min(max(2, trn_sites), embs_trn.shape[1]))
    pca_trn.fit(embs_trn)
    pca_embs_trn = pca_trn.transform(embs_trn)

    if ft_path is not None:
        ft_dict = torch.load(ft_path, map_location='cpu')
        embs_ft = np.zeros((ft_sites, ft_dict[0]['site_model_state']['embedding']))
        for i in range(ft_sites):
            embs_ft[i] = ft_dict[i]['site_model_state']['embedding']
        pca_embs_ft = pca_trn.transform(embs_ft)
    else:
        pca_embs_ft = None

    site_per_deg = 10 if ft_path is None else 5
    if deg == 'mixed':
        colors = ['red', 'blue']
        ft_colors = ['lightcoral', 'cyan']
        labels = ['Gaussian noise ', 'class skew ']
        title = '10 noisy sites, 10 class skewed sites'
    elif deg == '3mixed':
        colors = ['red', 'blue', 'green']
        ft_colors = ['lightcoral', 'cyan', 'mediumspringgreen']
        labels = ['Gaussian noise ', 'class skew ', 'colorjitter ']
        title = '10 noisy sites, 10 class skewed sites, 10 colorjitter sites'
    elif deg == 'jittermix':
        colors = ['red', 'blue', 'green', 'orange']
        ft_colors = ['lightcoral', 'cyan', 'mediumspringgreen', 'khaki']
        labels = ['brightness ', 'contrast ', 'saturation ', 'hue ']
        title = '10 brightness sites, 10 contrast sites, 10 saturation sites, 10 hue sites'
    matplotlib.pyplot.set_loglevel (level = 'info')
    fig, ax = plt.subplots(2)
    fig.set_size_inches(8,10)
    
    v = pca_embs_trn
    x = v[:,x_dim]
    y = v[:,y_dim]
    for i in range(len(colors)):
        ax[0].scatter(x[site_per_deg*i:site_per_deg*(i+1)], y[site_per_deg*i:site_per_deg*(i+1)], c=colors[i], label=labels[i]+'trn')
    diam_0, cent_x_0, cent_y_0 = max(x.max()-x.min(), y.max()-y.min()), x.mean(), y.mean()

    if ft_path is not None:
        v = pca_embs_ft
        x = v[:,x_dim]
        y = v[:,y_dim]
        for i in range(len(ft_colors)):
            ax[0].scatter(x[site_per_deg*i:site_per_deg*(i+1)], y[site_per_deg*i:site_per_deg*(i+1)], c=ft_colors[i], label=labels[i]+'ft')
    diam_1, cent_x_1, cent_y_1 = max(x.max()-x.min(), y.max()-y.min()), x.mean(), y.mean()

    diam, cent_x, cent_y = max(diam_0, diam_1), (cent_x_0 + cent_x_1) / 2, (cent_y_0 + cent_y_1) / 2
    ax[0].set_xlim(cent_x - diam, cent_x + diam)
    ax[0].set_ylim(cent_y - diam, cent_y + diam)
    ax[0].set_title(title_str + title + ' PCA')
    if legend:
        ax[0].legend()

    
    
    v = tsne_emb_trn
    x = v[:,0]
    y = v[:,1]
    for i in range(len(colors)):
        ax[1].scatter(x[site_per_deg*i:site_per_deg*(i+1)], y[site_per_deg*i:site_per_deg*(i+1)], c=colors[i], label=labels[i]+'trn')
    diam, cent_x, cent_y = max(x.max()-x.min(), y.max()-y.min()), x.mean(), y.mean()
    ax[1].set_xlim(cent_x - diam, cent_x + diam)
    ax[1].set_ylim(cent_y - diam, cent_y + diam)
    ax[1].set_title(title_str + title + ' TSNE')
    if legend:
        ax[1].legend()

    return tsne_emb_trn, pca_embs_trn, pca_embs_ft, pca_trn