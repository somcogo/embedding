import os
import random
import glob

import numpy as np
import torch

from config import get_new_config
from main import new_main_plus_ft

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.set_num_threads(8)

dataset = 'digits'

if dataset == 'digits':
    site = 40
    fts = 8
    deg = ['digits']
elif dataset == 'cifar10' or dataset == 'cifar100':
    site = 30
    fts = 6
    deg = ['addgauss', 'colorjitter', 'alphascale']

comment = 'linear'
logdir = 'test'
comm_rounds = 1000
norm_layer = 'in'
no_batch_running_stats = True
cl_per_site = None

emb_dim = 32
emb_vec = None

strategy = 'pureemb'
model_type = 'embbn4'
fed_prox = 0
feature_dim = 64
trn_set_size = None

main_fn = new_main_plus_ft

only_ft = False
fed_bn_emb_ft = False
state_dict = None
trn_state_dict = None

cross_val_id = 0
gl_seed = 0

torch.manual_seed(gl_seed)
random.seed(gl_seed)
np.random.seed(gl_seed)

def run():
    config = get_new_config(logdir, comment, site, deg, comm_rounds, strategy, model_type, fed_prox, emb_dim=emb_dim, ft_site_number=fts, cross_val_id=cross_val_id, gl_seed=gl_seed, norm_layer=norm_layer, no_batch_running_stats=no_batch_running_stats, ft_emb_vec=emb_vec, cl_per_site=cl_per_site, dataset=dataset, feature_dim=feature_dim, trn_set_size=trn_set_size, fed_bn_emb_ft=fed_bn_emb_ft)
    main_fn(state_dict=state_dict, trn_state_dict=trn_state_dict, only_ft=only_ft, **config)

if __name__ == '__main__':
    run()