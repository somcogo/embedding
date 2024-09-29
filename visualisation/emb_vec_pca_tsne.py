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
    
    pca_trn = PCA(n_components=min(max(2, trn_sites), embs_trn.shape[1]))
    pca_trn.fit(embs_trn)
    pca_embs_trn = pca_trn.transform(embs_trn)

    if ft_path is not None:
        ft_dict = torch.load(ft_path, map_location='cpu')
        embs_ft = np.zeros((ft_sites, ft_dict[0]['site_model_state']['embedding'].shape[0]))
        for i in range(ft_sites):
            embs_ft[i] = ft_dict[i]['site_model_state']['embedding']
        embs_comb = np.concatenate([embs_trn, embs_ft], axis=0)
        pca_embs_ft = pca_trn.transform(embs_ft)
        pca_embs_comb = np.concatenate([pca_embs_trn, pca_embs_ft], axis=0)
    else:
        pca_embs_ft = None
        embs_comb = embs_trn
        pca_embs_comb = pca_embs_trn

    tsne_emb_comb = TSNE(perplexity=min(5, trn_sites)).fit_transform(embs_comb)

    if deg == 'mixed':
        colors = ['red', 'blue']
        ft_colors = ['lightcoral', 'cyan']
        labels = ['Gaussian noise ', 'class skew ']
        title = ''
        # title = '10 noisy sites, 10 class skewed sites'
        site_per_deg = trn_sites // 2
        ft_site_per_deg = ft_sites // 2
    elif deg == '3mixed':
        colors = ['red', 'blue', 'green']
        ft_colors = ['lightcoral', 'cyan', 'mediumspringgreen']
        labels = ['Gaussian noise ', 'class skew ', 'colorjitter ']
        # title = '10 noisy sites, 10 class skewed sites, 10 colorjitter sites'
        title = ''
        site_per_deg = trn_sites // 3
        ft_site_per_deg = ft_sites // 3
    elif deg == 'adcoal' or deg == 'adjial':
        colors = ['red', 'blue', 'green']
        ft_colors = ['lightcoral', 'cyan', 'mediumspringgreen']
        labels = ['Noise ', 'Colorjitter ', 'Class imbalance ']
        # title = '10 noisy sites, 10 class skewed sites, 10 colorjitter sites'
        title = ''
        site_per_deg = trn_sites // 3
        ft_site_per_deg = ft_sites // 3
    elif deg == 'jittermix':
        colors = ['red', 'blue', 'green', 'orange']
        ft_colors = ['lightcoral', 'cyan', 'mediumspringgreen', 'khaki']
        labels = ['brightness ', 'contrast ', 'saturation ', 'hue ']
        # title = '10 brightness sites, 10 contrast sites, 10 saturation sites, 10 hue sites'
        title = ''
        site_per_deg = trn_sites // 4
        ft_site_per_deg = ft_sites // 4
    elif deg == 'digits':
        colors = ['red', 'blue', 'green', 'orange']
        ft_colors = ['lightcoral', 'cyan', 'mediumspringgreen', 'khaki']
        labels = ['mnist ', 'usps ', 'svhn ', 'syn ']
        # title = '10 brightness sites, 10 contrast sites, 10 saturation sites, 10 hue sites'
        title = ''
        site_per_deg = trn_sites // 4
        ft_site_per_deg = ft_sites // 4
    else:
        colors = ['red']
        ft_colors = ['lightcoral']
        labels = ['']
        title = ''
        site_per_deg = trn_sites
        ft_site_per_deg = ft_sites
        
    matplotlib.pyplot.set_loglevel (level = 'info')
    fig, ax = plt.subplots(2)
    fig.set_size_inches(8,10)
    
    v = pca_embs_trn
    x = v[:, x_dim]
    y = v[:, y_dim]
    for i in range(len(colors)):
        ax[0].scatter(x[site_per_deg*i:site_per_deg*(i+1)], y[site_per_deg*i:site_per_deg*(i+1)], c=colors[i], label=labels[i]+'trn')

    if ft_path is not None:
        v = pca_embs_ft
        x = v[:, x_dim]
        y = v[:, y_dim]
        for i in range(len(ft_colors)):
            ax[0].scatter(x[ft_site_per_deg*i:ft_site_per_deg*(i+1)], y[ft_site_per_deg*i:ft_site_per_deg*(i+1)], c=ft_colors[i], label=labels[i]+'ft')
    diam = max(pca_embs_comb[:, x_dim].max() - pca_embs_comb[:, x_dim].min(), pca_embs_comb[:, y_dim].max() - pca_embs_comb[:, y_dim].min())
    cent_x = pca_embs_comb[:, x_dim].mean()
    cent_y = pca_embs_comb[:, y_dim].mean()
    
    ax[0].set_xlim(cent_x - diam, cent_x + diam)
    ax[0].set_ylim(cent_y - diam, cent_y + diam)
    ax[0].set_title(title_str + title + ' PCA')
    if legend:
        ax[0].legend()

    
    
    v = tsne_emb_comb
    x = v[:, 0]
    y = v[:, 1]
    for i in range(len(colors)):
        ax[1].scatter(x[site_per_deg*i:site_per_deg*(i+1)], y[site_per_deg*i:site_per_deg*(i+1)], c=colors[i], label=labels[i]+'trn')
    if ft_path is not None:
        for i in range(len(ft_colors)):
            ax[1].scatter(x[trn_sites + ft_site_per_deg*i:trn_sites + ft_site_per_deg*(i+1)], y[trn_sites + ft_site_per_deg*i:trn_sites + ft_site_per_deg*(i+1)], c=ft_colors[i], label=labels[i]+'ft')
    diam, cent_x, cent_y = max(x.max()-x.min(), y.max()-y.min()), x.mean(), y.mean()
    ax[1].set_xlim(cent_x - diam, cent_x + diam)
    ax[1].set_ylim(cent_y - diam, cent_y + diam)
    ax[1].set_title(title_str + title + ' TSNE')
    if legend:
        ax[1].legend()

    return tsne_emb_comb, pca_embs_trn, pca_embs_ft, pca_trn