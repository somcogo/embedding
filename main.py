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
        self.trainer = self.initTrainer()
        self.fine_tuner = self.initFineTuner()

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

    def initTrainer(self):
        self.trainer = EmbeddingTraining(epochs=None, logdir=None, lr=None, comment=None, site_number=self.trn_site_number, model_name=None, model_type=None, optimizer_type=None, scheduler_mode=None, T_max=None, save_model=None, strategy=None, finetuning=None, sites=self.sites[:self.trn_site_number])

    def initFineTuner(self):
        pass

    def main(self):
        trained_model, training_metrics = self.trainer.train()
        fine_tuning_metrics = self.fine_tuner.train()