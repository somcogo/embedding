import os
import random

import numpy as np
import torch

from config import get_config, get_new_config
from main import main, new_main, new_main_plus_ft
from utils.val_and_vis import eval_points, get_points, load_embs

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_num_threads(8)

site = 40
fts = 8
deg = ['digits']
comment = 'size2res20ftpca20trnembs'
# comment = 'ftembeval'
logdir = '4digits'
comm_rounds = 1000
norm_layer = 'bn'
no_batch_running_stats = True
cl_per_site = 4

emb_dim = 64

strategy = 'pureemb'
model_type = 'embbn4'
fed_prox = 0
prox_map = False
ncc_lambda = 0.

main_fn = new_main_plus_ft

model_path = 'saved_models/4digits/2024_08_20-12_04_49-edim-pureemb-classification-resnet18-embbn4-di-s40-fts8-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4.state'
cross_val_id = None
gl_seed = 0

p_mode = 'grid'
saved_embs = load_embs('saved_models/4digits/2024_08_20-12_04_49-edim-pureemb-classification-resnet18-embbn4-di-s40-fts8-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4.state')
# saved_ft_embs = load_embs('saved_models/4digits_ft/2024_08_21-00_36_28-edim-pureemb-classification-resnet18-embbn4-di-s40-fts8-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4-onlyemb.state')
vectors = np.zeros((site-fts, emb_dim))
for i in range(site - fts):
    vectors[i] = saved_embs[i]
# for i in range(site - fts, site):
#     vectors[i] = saved_ft_embs[i - site + fts]

torch.manual_seed(gl_seed)
random.seed(gl_seed)
np.random.seed(gl_seed)

def run():
    config = get_new_config(logdir, comment, site, deg, comm_rounds, strategy, model_type, fed_prox, prox_map, emb_dim=emb_dim, ft_site_number=fts, cross_val_id=cross_val_id, gl_seed=gl_seed, norm_layer=norm_layer, no_batch_running_stats=no_batch_running_stats, cl_per_site=cl_per_site, ncc_lambda=ncc_lambda)
    points, pca = get_points(model_path, points=p_mode, vectors=vectors, **config)
    losses, accuracies, embeddings, cl = eval_points(embeddings=points, model_path=model_path, device='cuda', **config)
    save_comm = config['comment']
    save_dir = os.path.join('loss_vis', logdir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'losses':losses,
                'accuracies':accuracies,
                'embeddings':embeddings,
                'classes':cl,
                'pca':pca},
                os.path.join(save_dir, save_comm))

if __name__ == '__main__':
    run()