import os

import torch
import numpy as np

from utils.data_loader import refactored_get_dls
from utils.ops import get_test_transforms, refactored_get_transforms, refactored_get_ft_indices
from training import EmbeddingTraining
from utils.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

def new_main_plus_ft(logdir, comment, degradation, site_number, data_part_seed, transform_gen_seed, tr_config, ft_strategy, state_dict, trn_state_dict, ft_site_number, only_ft, ft_emb_lr, cross_val_id, gl_seed, ft_emb_vec, cl_per_site, trn_set_size, **config):
    log.info(comment)

    trn_dl_list, val_dl_list = refactored_get_dls(dataset=config['dataset'], batch_size=config['batch_size'], degs=degradation, n_sites=site_number, seed=data_part_seed, cross_val_id=cross_val_id, gl_seed=gl_seed, cl_per_site=cl_per_site, trn_set_size=trn_set_size)
    transform_list = refactored_get_transforms(site_number=site_number, seed=transform_gen_seed, degs=degradation, device='cuda' if torch.cuda.is_available() else 'cpu', **tr_config)
    if degradation == ['digits']:
        site_degs = np.repeat(np.arange(4), site_number // 4)
    else:
        site_per_deg = site_number // len(degradation) if type(degradation) == list else site_number
        site_degs = np.repeat(np.arange(site_number // site_per_deg), site_per_deg)
    site_dict = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'transform': transform_list[ndx],
                    'deg':site_degs[ndx]}
                    for ndx in range(site_number)]

    ft_indices = refactored_get_ft_indices(site_number, ft_site_number, degradation)
    trn_site_dict = [site_dict[i] for i in range(len(site_dict)) if i not in ft_indices]
    ft_site_dict = [site_dict[i] for i in range(len(site_dict)) if i in ft_indices]
    
    res = {}
    if site_number > 0 and not only_ft:
        trainer = EmbeddingTraining(logdir=logdir, comment=comment, sites=trn_site_dict, state_dict=trn_state_dict, **config)
        acc, state_dict = trainer.train()
        res['trn'] = acc

    if ft_site_number > 0:
        ft_comment = comment + '-' + ft_strategy
        ft_logdir = logdir + '_ft'
        config['comm_rounds'] = 200
        config['T_max'] = 200
        trn_strategy = config['strategy']
        config['strategy'] = ft_strategy
        config['embedding_lr'] = ft_emb_lr
        ft_trainer = EmbeddingTraining(logdir=ft_logdir, comment=ft_comment, state_dict=state_dict, sites=ft_site_dict, finetuning=True, ft_emb_vec=ft_emb_vec, **config)
        ft_acc, ft_state_dict = ft_trainer.train()
        res['ft'] = ft_acc
    
    if len(res.keys()) > 0:
        save_path = os.path.join('path/to/results', logdir)
        os.makedirs(save_path, exist_ok=True)
        res['strats'] = [trn_strategy, ft_strategy]
        torch.save(res, os.path.join(save_path, comment + '.pt'))