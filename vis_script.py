import os
import random

import numpy as np
import torch

from config import get_config, get_new_config
from main import main, new_main, new_main_plus_ft
from utils.val_and_vis import eval_points, get_points

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
torch.set_num_threads(8)

site = 10
fts = 2
deg = 'classshard'
comment = 'pcagrid182x100100'
logdir = 'test'
comm_rounds = 1000

emb_dim = 4

strategy = 'noembed'
model_type = 'embbn4'
fed_prox = 0
prox_map = False

main_fn = new_main_plus_ft

model_path = 'saved_models/seededruns/2024_07_14-00_56_24-fedbn-pureemb-classification-resnet18-embbn4-classshard-s10-fts2-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.001-embdim-4-cifar10-fedp-0-proxm-False-xvalNone-gls0.state'
cross_val_id = None
gl_seed = 0

p_mode = 'grid'

torch.manual_seed(gl_seed)
random.seed(gl_seed)
np.random.seed(gl_seed)

def run():
    config = get_new_config(logdir, comment, site, deg, comm_rounds, strategy, model_type, fed_prox, prox_map, emb_dim=emb_dim, ft_site_number=fts, cross_val_id=cross_val_id, gl_seed=gl_seed)
    points = get_points(model_path, points=p_mode, **config)
    losses, embeddings = eval_points(embeddings=points, model_path=model_path, device='cuda', **config)
    save_comm = config['comment']
    save_dir = os.path.join('loss_vis', logdir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'losses':losses,
                'embeddings':embeddings},
                os.path.join(save_dir, save_comm))

if __name__ == '__main__':
    run()