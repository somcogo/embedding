import os

import torch
import numpy as np

from utils.data_loader import get_dl_lists, new_get_dl_lists, refactored_get_dls
from utils.ops import getTransformList, get_class_list, get_test_transforms, get_ft_indices, refactored_get_transforms, refactored_get_ft_indices
from training import EmbeddingTraining
from utils.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def main(logdir, comment, task, model_name, model_type, degradation,
         site_number, trn_site_number, embedding_dim, batch_size,
         cross_val_id, comm_rounds, ft_comm_rounds=None, iterations=50,
         lr=None, ff_lr=None, emb_lr=None, ft_lr=None, ft_ff_lr=None, ft_emb_lr=None, optimizer_type=None,
         weight_decay=None, scheduler_mode=None, ft_scheduler='cosine', T_max=None, save_model=None,
         strategy=None, ft_strategies=None, cifar=True, data_part_seed=0,
         transform_gen_seed=1, model_path=None, dataset=None, alpha=1e7,
         tr_config=None, partition='dirichlet', feature_dims=None,
         label_smoothing=0., trn_logging=True):
    save_path = os.path.join('/home/hansel/developer/embedding/results', logdir)
    os.makedirs(save_path, exist_ok=True)
    log.info(comment)
    
    assert task in ['classification', 'segmentation']
    assert site_number >= trn_site_number

    trn_dl_list, val_dl_list = get_dl_lists(dataset, batch_size, partition=partition, n_site=site_number, alpha=alpha, seed=data_part_seed, use_hdf5=True, cross_val_id=cross_val_id)
    transform_list = getTransformList(degradation, site_number, seed=transform_gen_seed, device='cuda' if torch.cuda.is_available() else 'cpu', **tr_config)
    class_list = get_class_list(task=task, site_number=site_number, class_number=18 if dataset == 'celeba' else None, class_seed=2, degradation=degradation)
    sites = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'transform': transform_list[ndx],
                    'classes': class_list[ndx]}
                    for ndx in range(site_number)]
    
    trainer = EmbeddingTraining(comm_rounds=comm_rounds, logdir=logdir, lr=lr, ffwrd_lr=ff_lr, embedding_lr=emb_lr, weight_decay=weight_decay, comment=comment, dataset=dataset, site_number=trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model, strategy=strategy, finetuning=False, embed_dim=embedding_dim, sites=sites[:trn_site_number], cifar=cifar, model_path=model_path, iterations=iterations, task=task, feature_dims=feature_dims, label_smoothing=label_smoothing, trn_logging=trn_logging)

    if ft_strategies is not None and len(ft_strategies) > 0 and site_number > trn_site_number:
        ft_trainers = []
        logdir = os.path.join(logdir, 'finetuning')
        for strategy in ft_strategies:
            str_comment = comment + '-' + strategy
            ft_trainers.append(EmbeddingTraining(comm_rounds=ft_comm_rounds, logdir=logdir, lr=ft_lr, ffwrd_lr=ft_ff_lr, embedding_lr=ft_emb_lr, weight_decay=weight_decay, comment=str_comment, dataset=dataset, site_number=site_number - trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=ft_scheduler, save_model=save_model, strategy=strategy, finetuning=True, embed_dim=embedding_dim, sites=sites[trn_site_number:], cifar=cifar, iterations=iterations, task=task, feature_dims=feature_dims, label_smoothing=label_smoothing, trn_logging=trn_logging))
    else:
        ft_trainers = None
    
    training_metrics, state_dict = trainer.train()
    results = {'training':training_metrics}
    if ft_trainers is not None and len(ft_trainers) > 0:
        fine_tuning_metrics = {}
        for trainer in ft_trainers:
            fine_tuning_metrics[trainer.strategy] = trainer.train(state_dict)[0]
        results['fine_tuning'] = fine_tuning_metrics

    torch.save(results, os.path.join(save_path, comment))
    if cross_val_id is not None:
        return results
    

def ft_main(logdir, comment, task, model_name, model_type, degradation,
         site_number, trn_site_number, embedding_dim, batch_size,
         cross_val_id, ft_comm_rounds=None, iterations=50,
         lr=None, ff_lr=None, emb_lr=None, optimizer_type=None,
         weight_decay=None, ft_scheduler='cosine', save_model=None,
         strategy=None, cifar=True, data_part_seed=0,
         transform_gen_seed=1, dataset=None, alpha=1e7,
         tr_config=None, partition='dirichlet', feature_dims=None,
         label_smoothing=0., trn_logging=True, state_dict=None):
    save_path = os.path.join('/home/hansel/developer/embedding/results', logdir)
    os.makedirs(save_path, exist_ok=True)
    log.info(comment)
    
    assert task in ['classification', 'segmentation']
    assert site_number >= trn_site_number

    trn_dl_list, val_dl_list = get_dl_lists(dataset, batch_size, partition=partition, n_site=site_number, alpha=alpha, seed=data_part_seed, use_hdf5=True, cross_val_id=cross_val_id)
    transform_list = getTransformList(degradation, site_number, seed=transform_gen_seed, device='cuda' if torch.cuda.is_available() else 'cpu', **tr_config)
    class_list = get_class_list(task=task, site_number=site_number, class_number=18 if dataset == 'celeba' else None, class_seed=2, degradation=degradation)
    sites = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'transform': transform_list[ndx],
                    'classes': class_list[ndx]}
                    for ndx in range(site_number)]
    
    ft_trainer = EmbeddingTraining(comm_rounds=ft_comm_rounds, logdir=logdir, lr=lr, ffwrd_lr=ff_lr, embedding_lr=emb_lr, weight_decay=weight_decay, comment=comment, dataset=dataset, site_number=site_number - trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=ft_scheduler, save_model=save_model, strategy=strategy, finetuning=True, embed_dim=embedding_dim, sites=sites[trn_site_number:], cifar=cifar, iterations=iterations, task=task, feature_dims=feature_dims, label_smoothing=label_smoothing, trn_logging=trn_logging)
    
    fine_tuning_metrics = ft_trainer.train(state_dict)[0]
    results = fine_tuning_metrics

    torch.save(results, os.path.join(save_path, comment))
    if cross_val_id is not None:
        return results


def new_main(logdir, comment, degradation, site_number, data_part_seed, transform_gen_seed, tr_config, **config):
    save_path = os.path.join('/home/hansel/developer/embedding/results', logdir)
    os.makedirs(save_path, exist_ok=True)
    log.info(comment)

    trn_dl_list, val_dl_list = new_get_dl_lists(dataset=config['dataset'], batch_size=config['batch_size'], degradation=degradation, n_site=site_number, seed=data_part_seed)
    transform_list = get_test_transforms(site_number=site_number, seed=transform_gen_seed, degradation=degradation, device='cuda' if torch.cuda.is_available() else 'cpu', **tr_config)
    class_list = get_class_list(task='classification', site_number=site_number, class_number=18 if config['dataset'] == 'celeba' else None, class_seed=2, degradation=degradation)
    site_dict = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'transform': transform_list[ndx],
                    'classes': class_list[ndx]}
                    for ndx in range(site_number)]
    
    ft_trainer = EmbeddingTraining(logdir=logdir, comment=comment, site_number=site_number, sites=site_dict, **config)
    ft_trainer.train()

def new_main_plus_ft(logdir, comment, degradation, site_number, data_part_seed, transform_gen_seed, tr_config, ft_strategy, state_dict, ft_site_number, only_ft, ft_emb_lr, cross_val_id, gl_seed, ft_emb_vec, cl_per_site, **config):
    save_path = os.path.join('/home/hansel/developer/embedding/results', logdir)
    os.makedirs(save_path, exist_ok=True)
    log.info(comment)

    trn_dl_list, val_dl_list = refactored_get_dls(dataset=config['dataset'], batch_size=config['batch_size'], degs=degradation, n_sites=site_number, seed=data_part_seed, cross_val_id=cross_val_id, gl_seed=gl_seed, cl_per_site=cl_per_site, alpha=config['alpha'])
    transform_list = refactored_get_transforms(site_number=site_number, seed=transform_gen_seed, degs=degradation, device='cuda' if torch.cuda.is_available() else 'cpu', **tr_config)
    class_list = get_class_list(task='classification', site_number=site_number, class_number=18 if config['dataset'] == 'celeba' else None, class_seed=2, degradation=degradation)
    if degradation == ['digits']:
        site_degs = np.repeat(np.arange(4), site_number // 4)
    else:
        site_per_deg = site_number // len(degradation) if type(degradation) == list else site_number
        site_degs = np.repeat(np.arange(site_number // site_per_deg), site_per_deg)
    site_dict = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'transform': transform_list[ndx],
                    'classes': class_list[ndx],
                    'deg':site_degs[ndx]}
                    for ndx in range(site_number)]

    ft_indices = refactored_get_ft_indices(site_number, ft_site_number, degradation)
    trn_site_dict = [site_dict[i] for i in range(len(site_dict)) if i not in ft_indices]
    ft_site_dict = [site_dict[i] for i in range(len(site_dict)) if i in ft_indices]
    
    if site_number > 0 and not only_ft:
        trainer = EmbeddingTraining(logdir=logdir, comment=comment, sites=trn_site_dict, **config)
        acc, state_dict = trainer.train()

    if ft_site_number > 0:
        ft_comment = comment + '-' + ft_strategy
        ft_logdir = logdir + '_ft'
        config['comm_rounds'] = 200
        config['T_max'] = 200
        config['strategy'] = ft_strategy
        config['embedding_lr'] = ft_emb_lr
        ft_trainer = EmbeddingTraining(logdir=ft_logdir, comment=ft_comment, state_dict=state_dict, sites=ft_site_dict, finetuning=True, ft_emb_vec=ft_emb_vec, **config)
        ft_trainer.train()