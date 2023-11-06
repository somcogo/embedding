import os

import torch

from utils.data_loader import get_dl_lists
from utils.ops import getTransformList
from training import EmbeddingTraining

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
                 epochs,
                 ft_epochs=None,
                 lr=None,
                 optimizer_type=None,
                 scheduler_mode=None,
                 T_max=None,
                 save_model=None,
                 strategy=None,
                 ft_strategies=None,
                 data_part_seed=0,
                 transform_gen_seed=1):
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
        self.save_path = os.path.join('./results', logdir)
        os.makedirs(self.save_path, exist_ok=True)

        assert self.task in ['classification', 'reconstruction', 'segmentation']
        assert self.site_number >= self.trn_site_number
        # assert self.model_name in ['internimage', 'resnet18', 'drunet']

        self.initVariables()
        self.initSites(data_part_seed, transform_gen_seed)
        self.initTrainer(epochs=epochs, logdir=logdir, lr=lr, comment=comment, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max,save_model=save_model, strategy=strategy)
        if site_number > trn_site_number:
            self.initFineTuner(epochs=ft_epochs, logdir=logdir, lr=lr, comment=comment, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model, strategies=ft_strategies)

    def initVariables(self):
        if self.task == 'classification':
            self.dataset = 'imagenet'
        elif self.task == 'reconstruction':
            self.dataset = 'imagenet'
        else:
            self. dataset = 'ade20k'

    def initSites(self, data_part_seed, transform_gen_seed):
        trn_dl_list, val_dl_list = get_dl_lists(self.dataset, self.batch_size, partition='dirichlet', n_site=self.site_number, alpha=1, k_fold_val_id=self.k_fold_val_id, seed=data_part_seed, use_hdf5=True)
        transform_list = getTransformList(self.degradation, self.site_number, seed=transform_gen_seed)
        site_dict = [{'trn_dl': trn_dl_list[ndx],
                     'val_dl': val_dl_list[ndx],
                     'transform': transform_list[ndx]}
                     for ndx in range(self.site_number)]
        self.sites = site_dict

    def initTrainer(self, epochs, logdir, lr, comment, model_name, model_type, optimizer_type, scheduler_mode, T_max, save_model, strategy):
        self.trainer = EmbeddingTraining(epochs=epochs, logdir=logdir, lr=lr, comment=comment, dataset=self.dataset, site_number=self.trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model, strategy=strategy, finetuning=False, sites=self.sites[:self.trn_site_number])

    def initFineTuner(self, epochs, logdir, lr, comment, model_name, model_type, optimizer_type, scheduler_mode, T_max, save_model, strategies):
        ft_trainers = []
        logdir = os.path.join(logdir, 'finetuning')
        for strategy in strategies:
            str_comment = comment + strategy
            ft_trainers.append(EmbeddingTraining(epochs=epochs, logdir=logdir, lr=lr, comment=str_comment, dataset=self.dataset, site_number=self.site_number-self.trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, save_model=save_model, strategy=strategy, finetuning=True, sites=self.sites[self.trn_site_number:]))
        self.ft_trainers = ft_trainers

    def main(self):
        training_metrics, state_dict = self.trainer.train()
        if self.site_number > self.trn_site_number:
            fine_tuning_metrics = {}
            for trainer in self.ft_trainers:
                fine_tuning_metrics[trainer.strategy] = trainer.train(state_dict)[0]

        results = {'training_metrics':training_metrics,
                   'fine_tuning_accuracies':fine_tuning_metrics}
        torch.save(results, os.path.join(self.save_path, self.comment))