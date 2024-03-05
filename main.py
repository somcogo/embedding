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

class EmbeddingTask:
    def __init__(self,
                 logdir,
                 comment,
                 task,
                 model_name,
                 model_type,
                 degradation,
                 site_number,
                 trn_site_number,
                 embedding_dim,
                 batch_size,
                 k_fold_val_id,
                 comm_rounds,
                 ft_comm_rounds=None,
                 iterations=50,
                 lr=None,
                 ff_lr=None,
                 emb_lr=None,
                 optimizer_type=None,
                 weight_decay=None,
                 scheduler_mode=None,
                 T_max=None,
                 save_model=None,
                 strategy=None,
                 ft_strategies=None,
                 cifar=True,
                 data_part_seed=0,
                 transform_gen_seed=1,
                 model_path=None,
                 extra_conv=False,
                 dataset=None,
                 alpha=1e7,
                 one_hot_emb=False,
                 emb_trn_cycle=False,
                 tr_config=None,
                 partition='dirichlet'):
        self.task = task
        self.model_name = model_name
        self.model_type = model_type
        self.degradation = degradation
        self.site_number = site_number
        self.trn_site_number = trn_site_number
        self.model_name = model_name
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.k_fold_val_id = k_fold_val_id
        self.comment = comment
        self.save_path = os.path.join('/home/hansel/developer/embedding/results', logdir)
        os.makedirs(self.save_path, exist_ok=True)
        log.info(comment)

        assert self.task in ['classification', 'reconstruction', 'segmentation']
        assert self.site_number >= self.trn_site_number
        # assert self.model_name in ['internimage', 'resnet18', 'drunet']

        self.initVariables(dataset)
        self.initSites(data_part_seed, transform_gen_seed, alpha=alpha, tr_config=tr_config, partition=partition)
        self.initTrainer(comm_rounds=comm_rounds, logdir=logdir, lr=lr, ff_lr=ff_lr, emb_lr=emb_lr, weight_decay=weight_decay, comment=comment, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max,save_model=save_model, strategy=strategy, cifar=cifar, model_path=model_path, extra_conv=extra_conv, iterations=iterations, one_hot_emb=one_hot_emb, emb_trn_cycle=emb_trn_cycle)
        if site_number > trn_site_number:
            self.initFineTuner(comm_rounds=ft_comm_rounds, logdir=logdir, lr=lr, ff_lr=ff_lr, emb_lr=emb_lr, weight_decay=weight_decay, comment=comment, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model, strategies=ft_strategies, cifar=cifar, iterations=iterations)

    def initVariables(self, dataset):
        if self.task == 'classification':
            self.dataset = 'imagenet' if dataset is None else dataset
        elif self.task == 'reconstruction':
            self.dataset = 'imagenet' if dataset is None else dataset
        else:
            self. dataset = 'celeba' if dataset is None else dataset

    def initSites(self, data_part_seed, transform_gen_seed, alpha, tr_config, partition):
        trn_dl_list, val_dl_list = get_dl_lists(self.dataset, self.batch_size, partition=partition, n_site=self.site_number, alpha=alpha, k_fold_val_id=self.k_fold_val_id, seed=data_part_seed, use_hdf5=True)
        transform_list = getTransformList(self.degradation, self.site_number, seed=transform_gen_seed, device='cuda' if torch.cuda.is_available() else 'cpu', **tr_config)
        class_list = get_class_list(task=self.task, site_number=self.site_number, class_number=18 if self.dataset == 'celeba' else None, class_seed=2)
        site_dict = [{'trn_dl': trn_dl_list[ndx],
                     'val_dl': val_dl_list[ndx],
                     'transform': transform_list[ndx],
                     'classes': class_list[ndx]}
                     for ndx in range(self.site_number)]
        self.sites = site_dict

    def initTrainer(self, comm_rounds, logdir, lr, ff_lr, emb_lr, weight_decay, comment, model_name, model_type, optimizer_type, scheduler_mode, T_max, save_model, strategy, cifar, model_path, extra_conv, iterations, one_hot_emb, emb_trn_cycle):
        self.trainer = EmbeddingTraining(comm_rounds=comm_rounds, logdir=logdir, lr=lr, ffwrd_lr=ff_lr, embedding_lr=emb_lr, weight_decay=weight_decay, comment=comment, dataset=self.dataset, site_number=self.trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model, strategy=strategy, finetuning=False, sites=self.sites[:self.trn_site_number], cifar=cifar, model_path=model_path, iterations=iterations, one_hot_emb=one_hot_emb, emb_trn_cycle=emb_trn_cycle, task=self.task)

    def initFineTuner(self, comm_rounds, logdir, lr, ff_lr, emb_lr, weight_decay, comment, model_name, model_type, optimizer_type, scheduler_mode, T_max, save_model, strategies, cifar, iterations):
        ft_trainers = []
        logdir = os.path.join(logdir, 'finetuning')
        for strategy in strategies:
            str_comment = comment + strategy
            ft_trainers.append(EmbeddingTraining(comm_rounds=comm_rounds, logdir=logdir, lr=lr, ffwrd_lr=ff_lr, embedding_lr=emb_lr, weight_decay=weight_decay, comment=str_comment, dataset=self.dataset, site_number=self.site_number-self.trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, save_model=save_model, strategy=strategy, finetuning=True, sites=self.sites[self.trn_site_number:], cifar=cifar, iterations=iterations, one_hot_emb=False, emb_trn_cycle=False, task=self.task))
        self.ft_trainers = ft_trainers

    def main(self):
        training_metrics, state_dict = self.trainer.train()
        if self.site_number > self.trn_site_number:
            fine_tuning_metrics = {}
            for trainer in self.ft_trainers:
                fine_tuning_metrics[trainer.strategy] = trainer.train(state_dict)[0]

        results = {'training':training_metrics,
                   'fine_tuning':fine_tuning_metrics}
        torch.save(results, os.path.join(self.save_path, self.comment))