import os
import random

import numpy as np
import torch

from config import get_config, get_new_config
from main import main, new_main, new_main_plus_ft
from utils.val_and_vis import eval_points, get_points, load_embs, new_vis

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_num_threads(8)

p_mode = 'xygrid'
save_comm = 'size1-5res40try2'
case = 'mixed'

if case == 'mixed':
    site = 30
    logdir = 'alpha'
elif case == 'digits':
    site = 40
    logdir = '4digits'
fts = site // 5
emb_dim = 64

model_path = 'saved_models/alpha/2024_08_24-14_05_56-ncc-pureemb-classification-resnet18-embbn4-adcoal-s30-fts6-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-in-rst-True-clpr4-ncc0.0.state'
# model_path = 'saved_models/alpha/2024_08_24-14_05_56-ncc-pureemb-classification-resnet18-embbn4-adcoal-s30-fts6-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-in-rst-True-clpr4-ncc0.0.state'

saved_embs = load_embs(model_path)
saved_ft_embs = load_embs('saved_models/alpha_ft/2024_08_25-02_21_23-ncc-pureemb-classification-resnet18-embbn4-adcoal-s30-fts6-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-in-rst-True-clpr4-ncc0.0-onlyemb.state')
# saved_embs = load_embs('saved_models/alpha/2024_08_24-14_05_56-ncc-pureemb-classification-resnet18-embbn4-adcoal-s30-fts6-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-in-rst-True-clpr4-ncc0.0.state')
# saved_ft_embs = load_embs('saved_models/alpha_ft/2024_08_25-02_21_23-ncc-pureemb-classification-resnet18-embbn4-adcoal-s30-fts6-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-in-rst-True-clpr4-ncc0.0-onlyemb.state')

def run():
    vectors = np.zeros((site, emb_dim))
    for i in range(site - fts):
        vectors[i] = saved_embs[i]
    for i in range(site - fts, site):
        vectors[i] = saved_ft_embs[i - site + fts]
    eval_vectors, pca = get_points(model_path, p_mode, vectors)
    losses, imgs, vectors = new_vis(case, model_path, eval_vectors)
    save_dir = os.path.join('loss_vis', logdir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'losses':losses,
                'imgs':imgs,
                'vectors':vectors,
                'pca':pca},
                os.path.join(save_dir, save_comm))

if __name__ == '__main__':
    run()