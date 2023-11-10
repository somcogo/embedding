import os
import datetime
import math
import copy

import torch
from torchvision.transforms import Resize
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, LBFGS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.logconf import logging
from utils.data_loader import get_dl_lists
from utils.ops import aug_image, perturb
from utils.merge_strategies import get_layer_list
from utils.get_model import get_model

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class EmbeddingTraining:
    def __init__(self, epochs=500, batch_size=128, logdir='test', lr=1e-3,
                 comment='dwlpt', dataset='cifar10', site_number=1,
                 model_name='resnet18emb', optimizer_type='newadam',
                 scheduler_mode='cosine', pretrained=False, T_max=500,
                 label_smoothing=0.0, save_model=False, partition='regular',
                 alpha=None, strategy='noembed', finetuning=False, embed_dim=2,
                 model_path=None, embedding_lr=None, ffwrd_lr=None,
                 layer_number=4, k_fold_val_id=None, seed=None,
                 site_indices=None, input_perturbation=False, use_hdf5=False,
                 conv1_residual=True, fc_residual=True, colorjitter=False,
                 sites=None, model_type=None, weight_decay=1e-5):

        # self.settings = copy.deepcopy(locals())
        # del self.settings['self']
        # log.info(self.settings)
        log.info(comment)
        self.epochs = epochs
        self.logdir_name = logdir
        self.comment = comment
        self.dataset = dataset
        self.site_number = site_number
        self.model_name = model_name
        self.optimizer_type = optimizer_type
        self.scheduler_mode = scheduler_mode
        self.pretrained = pretrained
        if T_max is None or epochs > T_max:
            self.T_max = epochs
        else:
            self.T_max = T_max
        self.label_smoothing = label_smoothing
        self.save_model = save_model
        self.strategy = strategy
        self.finetuning = finetuning
        self.embed_dim = embed_dim
        self.model_path = model_path
        self.input_perturbation = input_perturbation
        if site_indices is None:
            site_indices = range(site_number)
        self.site_indices = site_indices
        self.use_hdf5 = use_hdf5
        self.conv1_residual = conv1_residual
        self.fc_residual = fc_residual
        self.colorjitter = colorjitter
        self.time_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.logdir_name)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        if sites is not None:
            self.trn_dls = [site['trn_dl'] for site in sites]
            self.val_dls = [site['val_dl'] for site in sites]
            self.transforms = [site['transform'] for site in sites]
            self.site_number = len(sites)
        else:
            self.trn_dls, self.val_dls = self.initDls(batch_size=batch_size, partition=partition, alpha=alpha, k_fold_val_id=k_fold_val_id, seed=seed, site_indices=site_indices)
            self.site_number = len(site_indices)
        self.models = self.initModels(embed_dim=embed_dim, layer_number=layer_number, model_type=model_type)
        self.optims = self.initOptimizers(lr, finetuning, weight_decay=weight_decay, embedding_lr=embedding_lr, ffwrd_lr=ffwrd_lr)
        self.schedulers = self.initSchedulers()
        assert len(self.trn_dls) == self.site_number and len(self.val_dls) == self.site_number and len(self.models) == self.site_number and len(self.optims) == self.site_number

    def initModels(self, embed_dim, layer_number, model_type):
        models, self.num_classes = get_model(self.dataset, self.model_name, self.site_number, embed_dim, layer_number, self.pretrained, conv1_residual=self.conv1_residual, fc_residual=self.fc_residual, model_type=model_type)

        if 'embedding.weight' in '\t'.join(models[0].state_dict().keys()) or hasattr(models[0], 'embedding'):
            if embed_dim > 2:
                self.mu_init = np.eye(self.site_number, embed_dim)
            else:
                mu_init = np.exp((2 * np.pi * 1j/ self.site_number)*np.arange(0,self.site_number))
                self.mu_init = np.stack([np.real(mu_init), np.imag(mu_init)], axis=1)
            if 'embedding.weight' in '\t'.join(models[0].state_dict().keys()):
                for i, model in enumerate(models):
                    init_weight = torch.from_numpy(self.mu_init[i]).repeat(self.site_number).reshape(self.site_number, embed_dim)
                    model.embedding.weight = nn.Parameter(init_weight)
            else:
                for i, model in enumerate(models):
                    init_weight = torch.from_numpy(self.mu_init[i])
                    model.embedding = nn.Parameter(init_weight)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            for i, model in enumerate(models):
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(self.device)
        return models

    def initOptimizers(self, lr, finetuning, weight_decay=None, embedding_lr=None, ffwrd_lr=None):
        optims = []
        for model in self.models:
            params_to_update = []
            if finetuning:
                assert self.strategy in ['finetuning', 'affinetoo', 'onlyfc', 'onlyemb']
                layer_list = get_layer_list(self.model_name, strategy=self.strategy, original_list=model.state_dict().keys())
                for name, param in model.named_parameters():
                    if name in layer_list:
                        params_to_update.append(param)
                    else:
                        param.requires_grad = False
            else:
                all_names = [name for name, _ in model.named_parameters()]
                embedding_names = []
                ffwrd_names = []
                if embedding_lr is not None:
                    embedding_names = [name for name in all_names if name.split('.')[0] == 'embedding']
                    params_to_update.append({'params':[param for name, param in model.named_parameters() if name in embedding_names], 'lr':embedding_lr})
                if ffwrd_lr is not None:
                    ffwrd_names = [name for name in all_names if 'generator' in name]
                    params_to_update.append({'params':[param for name, param in model.named_parameters() if name in ffwrd_names], 'lr':ffwrd_lr})
                params_to_update.append({'params':[param for name, param in model.named_parameters() if not name in embedding_names and not name in ffwrd_names]})

            if self.optimizer_type == 'adam':
                optim = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)
            elif self.optimizer_type == 'newadam':
                optim = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay, betas=(0.5,0.9))
            elif self.optimizer_type == 'adamw':
                optim = AdamW(params=params_to_update, lr=lr, weight_decay=weight_decay)
            elif self.optimizer_type == 'sgd':
                optim = SGD(params=params_to_update, lr=lr, weight_decay=weight_decay, momentum=0.9)
            elif self.optimizer_type == 'lbfgs':
                optim = LBFGS(params=params_to_update, lr=lr, line_search_fn='strong_wolfe')
            optims.append(optim)
        return optims
    
    def initSchedulers(self):
        if self.scheduler_mode is None:
            schedulers = None
        else:
            schedulers = []
            for optim in self.optims:
                if self.scheduler_mode == 'cosine':
                    scheduler = CosineAnnealingLR(optim, T_max=self.T_max, eta_min=1e-6)
                schedulers.append(scheduler)
            
        return schedulers

    def initDls(self, batch_size, partition, alpha, k_fold_val_id, seed, site_indices):
        if not self.finetuning:
            index_dict = torch.load('utils/index_maps_and_seeds/{}_saved_index_maps.pt'.format(self.dataset)) if partition in ['given', '5foldval'] else None
        else:
            index_dict = torch.load('utils/index_maps_and_seeds/{}_finetune.pt'.format(self.dataset)) if partition in ['given', '5foldval'] else None
        trn_idx_map = index_dict[self.site_number][alpha]['trn'] if index_dict is not None else None
        val_idx_map = index_dict[self.site_number][alpha]['val'] if index_dict is not None else None
        trn_dls, val_dls = get_dl_lists(dataset=self.dataset, partition=partition, n_site=self.site_number, batch_size=batch_size, alpha=alpha, net_dataidx_map_train=trn_idx_map, net_dataidx_map_test=val_idx_map, k_fold_val_id=k_fold_val_id, seed=seed, site_indices=site_indices, use_hdf5=self.use_hdf5)
        return trn_dls, val_dls

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.comment)

    def train(self, state_dict=None):
        log.info("Starting {}".format(type(self).__name__))
        self.mergeModels(is_init=True, model_path=self.model_path, state_dict=state_dict)

        trn_dls = self.trn_dls
        val_dls = self.val_dls

        saving_criterion = 0
        validation_cadence = 5

        if self.finetuning:
            val_metrics, accuracy, loss = self.doValidation(0, val_dls)
            self.logMetrics(0, 'val', val_metrics)
            log.info('Epoch {} of {}, accuracy/miou {}, val loss {}'.format(0, self.epochs, accuracy, loss))

        for epoch_ndx in range(1, self.epochs + 1):

            if epoch_ndx == 1:
                log.info("Epoch {} of {}, training on {} sites, using {} device".format(
                    epoch_ndx,
                    self.epochs,
                    len(trn_dls),
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trn_metrics = self.doTraining(epoch_ndx, trn_dls)
            self.logMetrics(epoch_ndx, 'trn', trn_metrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                val_metrics, accuracy, loss = self.doValidation(epoch_ndx, val_dls)
                self.logMetrics(epoch_ndx, 'val', val_metrics)
                saving_criterion = max(accuracy, saving_criterion)

                if self.save_model and accuracy==saving_criterion:
                    self.saveModel(epoch_ndx, val_metrics, trn_dls, val_dls)

                if epoch_ndx < 51 or epoch_ndx % 100 == 0:
                    log.info('Epoch {} of {}, accuracy {}, val loss {}'.format(epoch_ndx, self.epochs, accuracy, loss))            
            
            if self.scheduler_mode == 'cosine' and not self.finetuning:
                for scheduler in self.schedulers:
                    scheduler.step()
                    # log.debug(self.scheduler.get_last_lr())

            if self.site_number > 1 and not self.finetuning:
                self.mergeModels()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

        return saving_criterion, self.models[0].state_dict()

    def doTraining(self, epoch_ndx, trn_dls):
        for model in self.models:
            model.train()

        trn_metrics = torch.zeros(2 + 2*self.site_number + self.num_classes, device=self.device)
        loss = 0
        correct = 0
        total = 0
        correct_by_class = torch.zeros(self.num_classes, device=self.device)
        total_by_class = torch.zeros(self.num_classes, device=self.device)
        for ndx, trn_dl in enumerate(trn_dls):
            local_trn_metrics = torch.zeros(2 + 2*self.num_classes, len(trn_dl), device=self.device)

            for batch_ndx, batch_tuple in enumerate(trn_dl):
                # with torch.autograd.detect_anomaly():s
                    def closure():
                        self.optims[ndx].zero_grad()
                        loss, _ = self.computeBatchLoss(
                            batch_ndx,
                            batch_tuple,
                            self.models[ndx],
                            local_trn_metrics,
                            'trn',
                            ndx)
                        loss.backward()
                        return loss
                    # try:
                    if self.optimizer_type == 'lbfgs':
                        self.optims[ndx].step(closure)
                    else:
                        loss = closure()
                        self.optims[ndx].step()
                    # except:
                    

            loss += local_trn_metrics[-2].sum()
            correct += local_trn_metrics[-1].sum()
            total += len(trn_dl.dataset)

            correct_by_class += local_trn_metrics[:self.num_classes].sum(dim=1)
            total_by_class += local_trn_metrics[self.num_classes: 2*self.num_classes].sum(dim=1)

            trn_metrics[2*ndx] = local_trn_metrics[-2].sum() / len(trn_dl.dataset)
            trn_metrics[2*ndx + 1] = local_trn_metrics[-1].sum() / len(trn_dl.dataset)

        trn_metrics[2*self.site_number: 2*self.site_number + self.num_classes] = correct_by_class / total_by_class
        trn_metrics[-2] = loss / total
        trn_metrics[-1] = correct / total

        self.totalTrainingSamples_count += len(trn_dls[0].dataset)

        return trn_metrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dls):
        with torch.no_grad():
            for model in self.models:
                model.eval()
            if epoch_ndx == 1:
                log.warning('E{} Validation starting'.format(epoch_ndx))

            val_metrics = torch.zeros(2 + 2*self.site_number + self.num_classes, device=self.device)
            loss = 0
            correct = 0
            total = 0
            correct_by_class = torch.zeros(self.num_classes, device=self.device)
            total_by_class = torch.zeros(self.num_classes, device=self.device)
            for ndx, val_dl in enumerate(val_dls):
                local_val_metrics = torch.zeros(2 + 2*self.num_classes, len(val_dl), device=self.device)

                for batch_ndx, batch_tuple in enumerate(val_dl):
                    # with torch.autograd.detect_anomaly():
                        _, accuracy = self.computeBatchLoss(
                            batch_ndx,
                            batch_tuple,
                            self.models[ndx],
                            local_val_metrics,
                            'val',
                            ndx
                        )
                
                loss += local_val_metrics[-2].sum()
                correct += local_val_metrics[-1].sum()
                total += local_val_metrics[self.num_classes: 2*self.num_classes].sum()

                correct_by_class += local_val_metrics[:self.num_classes].sum(dim=1)
                total_by_class += local_val_metrics[self.num_classes: 2*self.num_classes].sum(dim=1)

                val_metrics[2*ndx] = local_val_metrics[-2].sum() / len(val_dl.dataset)
                val_metrics[2*ndx + 1] = local_val_metrics[-1].sum() / len(val_dl.dataset)

            val_metrics[2*self.site_number: 2*self.site_number + self.num_classes] = correct_by_class / total_by_class
            val_metrics[-2] = loss / total
            val_metrics[-1] = correct / total

        return val_metrics.to('cpu'), correct / total, loss / total

    def computeBatchLoss(self, batch_ndx, batch_tup, model, metrics, mode, site_id):
        batch, labels, img_id = batch_tup
        batch = batch.to(device=self.device, non_blocking=True).float().permute(0, 3, 1, 2)
        labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

        if self.input_perturbation:
            perturb_mode = 'colorjitter' if self.colorjitter else 'default'
            batch = perturb(batch, self.site_indices[site_id], self.device, perturb_mode)
        if mode == 'trn':
            batch = aug_image(batch, self.dataset)
        if self.model_name[:6] == 'maxvit':
            resize = Resize(224, antialias=True)
            batch = resize(batch)
        if hasattr(self, 'transforms'):
            batch = self.transforms[site_id](batch)

        if 'embedding.weight' in '\t'.join(model.state_dict().keys()):
            pred = model(batch, torch.tensor(site_id, device=self.device, dtype=torch.int))
        else:
            pred = model(batch)
        pred_label = torch.argmax(pred, dim=1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        loss = loss_fn(pred, labels)

        correct_mask = pred_label == labels
        correct = torch.sum(correct_mask)
        accuracy = correct / batch.shape[0] * 100

        for cls in range(self.num_classes):
            class_mask = labels == cls
            correct_in_cls = torch.sum(correct_mask[class_mask])
            total_in_cls = torch.sum(class_mask)
            metrics[cls, batch_ndx] = correct_in_cls
            metrics[self.num_classes + cls, batch_ndx] = total_in_cls


        metrics[-2, batch_ndx] = loss.detach()
        metrics[-1, batch_ndx] = correct

        return loss.sum(), accuracy

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics
    ):
        self.initTensorboardWriters()

        writer = getattr(self, mode_str + '_writer')
        for ndx in range(self.site_number):
            writer.add_scalar(
                'loss by site/site {}'.format(ndx),
                scalar_value=metrics[2*ndx],
                global_step=epoch_ndx
            )
            writer.add_scalar(
                'accuracy by site/site {}'.format(ndx),
                scalar_value=metrics[2*ndx + 1],
                global_step=epoch_ndx
            )
        for ndx in range(self.num_classes):
            writer.add_scalar(
                'accuracy by class/class {}'.format(ndx),
                scalar_value=metrics[2*self.site_number + ndx],
                global_step=epoch_ndx
            )
        writer.add_scalar(
            'loss/overall',
            scalar_value=metrics[-2],
            global_step=epoch_ndx
        )
        writer.add_scalar(
            'accuracy/overall',
            scalar_value=metrics[-1],
            global_step=epoch_ndx
        )
        writer.flush()

    def saveModel(self, epoch_ndx, val_metrics, trn_dls, val_dls):
        model_file_path = os.path.join(
            'saved_models',
            self.logdir_name,
            '{}-{}.state'.format(
                self.time_str,
                self.comment
            )
        )
        os.makedirs(os.path.dirname(model_file_path), mode=0o755, exist_ok=True)
        data_file_path = os.path.join(
            'saved_metrics',
            self.logdir_name,
            '{}-{}.state'.format(
                self.time_str,
                self.comment
            )
        )
        os.makedirs(os.path.dirname(data_file_path), mode=0o755, exist_ok=True)

        data_state = {'valmetrics':val_metrics.detach().cpu(),
                      'epoch': epoch_ndx,}
                    #   'settings':self.settings}
        model_state = {'epoch': epoch_ndx,}
        model_state['model_state'] = self.models[0].state_dict()
        layer_list = get_layer_list(model=self.model_name, strategy=self.strategy, original_list=[name for name, _ in self.models[0].named_parameters()])
        for ndx, model in enumerate(self.models):
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            data_state[ndx] = {
                'trn_labels': trn_dls[ndx].dataset.labels,
                'val_labels': val_dls[ndx].dataset.labels,
                'trn_indices': trn_dls[ndx].dataset.indices,
                'val_indices': val_dls[ndx].dataset.indices,
            }
            state_dict = model.state_dict()
            site_state_dict = {key: state_dict[key] for key in state_dict.keys() if key not in layer_list}
            model_state[ndx] = {
                'site_model_state': site_state_dict,
                'model_name': type(model).__name__,
                'optimizer_state': self.optims[ndx].state_dict(),
                'optimizer_name': type(self.optims[ndx]).__name__,
                'scheduler_state': self.schedulers[ndx].state_dict(),
                'scheduler_name': type(self.schedulers[ndx]).__name__,
                }
            if 'embedding.weight' in '\t'.join(model.state_dict().keys()):
                data_state[ndx]['emb_vector'] = model.state_dict()['embedding.weight'][ndx].detach().cpu()
            elif hasattr(model, 'embedding'):
                data_state[ndx]['emb_vector'] = model.embedding.detach().cpu()

        torch.save(model_state, model_file_path)
        torch.save(data_state, data_file_path)
        log.debug("Saved model params to {}".format(model_file_path))

    def mergeModels(self, is_init=False, model_path=None, state_dict=None):
        if is_init:
            if model_path is not None:
                loaded_dict = torch.load(model_path)
                if 'model_state' in loaded_dict.keys():
                    state_dict = loaded_dict['model_state']
                else:
                    state_dict = loaded_dict[0]['model_state']
            elif state_dict is None:
                state_dict = self.models[0].state_dict()
            if 'embedding.weight' in '\t'.join(state_dict.keys()):
                state_dict['embedding.weight'] = state_dict['embedding.weight'][0].unsqueeze(0).repeat(self.site_number, 1)
            for model in self.models:
                model.load_state_dict(state_dict, strict=False)
        else:
            original_list = [name for name, _ in self.models[0].named_parameters()]
            layer_list = get_layer_list(model=self.model_name, strategy=self.strategy, original_list=original_list)
            state_dicts = [model.state_dict() for model in self.models]
            param_dict = {layer: torch.zeros(state_dicts[0][layer].shape, device=self.device) for layer in layer_list}

            for layer in layer_list:
                for state_dict in state_dicts:
                    param_dict[layer] += state_dict[layer]
                param_dict[layer] /= len(state_dicts)

            for model in self.models:
                model.load_state_dict(param_dict, strict=False)


if __name__ == '__main__':
    EmbeddingTraining().train()
