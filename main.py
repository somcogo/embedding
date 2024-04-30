import os

import torch

from utils.data_loader import get_dl_lists
from utils.ops import getTransformList, get_class_list
from training import EmbeddingTraining
from utils.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def main(logdir, comment, task, model_name, model_type, degradation,
         site_number, trn_site_number, embedding_dim, batch_size,
         cross_val_id, comm_rounds, ft_comm_rounds=None, iterations=50,
         lr=None, ff_lr=None, emb_lr=None, optimizer_type=None,
         weight_decay=None, scheduler_mode=None, T_max=None, save_model=None,
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
    site_dict = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'transform': transform_list[ndx],
                    'classes': class_list[ndx]}
                    for ndx in range(site_number)]
    sites = site_dict
    
    trainer = EmbeddingTraining(comm_rounds=comm_rounds, logdir=logdir, lr=lr, ffwrd_lr=ff_lr, embedding_lr=emb_lr, weight_decay=weight_decay, comment=comment, dataset=dataset, site_number=trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model, strategy=strategy, finetuning=False, embed_dim=embedding_dim, sites=sites[:trn_site_number], cifar=cifar, model_path=model_path, iterations=iterations, task=task, feature_dims=feature_dims, label_smoothing=label_smoothing, trn_logging=trn_logging)

    if ft_strategies is not None and len(ft_strategies) > 0 and site_number > trn_site_number:
        ft_trainers = []
        logdir = os.path.join(logdir, 'finetuning')
        for strategy in ft_strategies:
            str_comment = comment + '-' + strategy
            ft_trainers.append(EmbeddingTraining(comm_rounds=ft_comm_rounds, logdir=logdir, lr=lr, ffwrd_lr=ff_lr, embedding_lr=emb_lr, weight_decay=weight_decay, comment=str_comment, dataset=dataset, site_number=site_number - trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, save_model=save_model, strategy=strategy, finetuning=True, embed_dim=embedding_dim, sites=sites[trn_site_number:], cifar=cifar, iterations=iterations, task=task, feature_dims=feature_dims, label_smoothing=label_smoothing, trn_logging=trn_logging))
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