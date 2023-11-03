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

        assert self.task in ['classification, reconstruction, segmentation']
        assert self.site_number >= self.trn_site_number
        assert self.model_name in ['internimage', 'resnet18', 'drunet']

        self.initVariables()
        self.sites = self.initSites(data_part_seed, transform_gen_seed)
        self.trainer = self.initTrainer(epochs=epochs, logdir=logdir, lr=lr, comment=comment, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max,save_model=save_model, strategy=strategy)
        self.fine_tuner = self.initFineTuner(epochs=ft_epochs, logdir=logdir, lr=lr, comment=comment, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model)

    def initVariables(self):
        if self.task == 'classification':
            self.dataset = 'tinyimagenet'
        elif self.task == 'reconstruction':
            self.dataset = 'tinyimagenet'
        else:
            self. dataset = 'ade20k'

    def initSites(self, data_part_seed, transform_gen_seed):
        trn_dl_list, val_dl_list = get_dl_lists(self.dataset, self.batch_size, partition='dirichlet', n_site=self.site_number, alpha=1e7, k_fold_val_id=self.k_fold_val_id, seed=data_part_seed, use_hdf5=True)
        transform_list = getTransformList(self.degradation, self.site_number, seed=transform_gen_seed)
        site_dict = [{'trn_dl': trn_dl_list[ndx],
                     'val_dl': val_dl_list[ndx],
                     'transform': transform_list[ndx]}
                     for ndx in range(self.site_number)]
        return site_dict

    def initTrainer(self, epochs, logdir, lr, comment, model_name, model_type, optimizer_type, scheduler_mode, T_max, save_model, strategy):
        self.trainer = EmbeddingTraining(epochs=epochs, logdir=logdir, lr=lr, comment=comment, site_number=self.trn_site_number, model_name=model_name, model_type=model_type, optimizer_type=optimizer_type, scheduler_mode=scheduler_mode, T_max=T_max, save_model=save_model, strategy=strategy, finetuning=False, sites=self.sites[:self.trn_site_number])

    def initFineTuner(self, epochs, logdir, lr, comment, model_name, model_type, optimizer_type, scheduler_mode, T_max, save_model, strategy):
        pass

    def main(self):
        trained_model, training_metrics = self.trainer.train()
        fine_tuning_metrics = self.fine_tuner.train()