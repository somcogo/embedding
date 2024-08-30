import os
import datetime
import math
import copy

import torch
from torchvision.transforms import Resize
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, LBFGS
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

from st_adam import Adam as ProxAdam
from utils.logconf import logging
from utils.data_loader import get_dl_lists
from utils.ops import transform_image, getTransformList, create_mask_from_onehot
from utils.merge_strategies import get_layer_list
from utils.get_model import get_model
from models.embedding_functionals import GeneralBatchNorm2d

torchvision.disable_beta_transforms_warning()
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class EmbeddingTraining:
    def __init__(self, comm_rounds=500, batch_size=128, logdir='test', lr=1e-3,
                 comment='dwlpt', dataset='cifar10', site_number=1,
                 model_name='resnet18emb', optimizer_type='newadam',
                 scheduler_mode='cosine', T_max=500, save_model=False,
                 partition='dirichlet', alpha=1e7, strategy='noembed',
                 finetuning=False, embed_dim=2, model_path=None,
                 embedding_lr=None, ffwrd_lr=None, k_fold_val_id=None, seed=0,
                 site_indices=None, task='classification', sites=None,
                 model_type=None, weight_decay=1e-5, cifar=True,
                 get_transforms=False, state_dict=None, iterations=None,
                 feature_dims=None, label_smoothing=0., trn_logging=True,
                 fed_prox=0., proximal_map=False, norm_layer='bn',
                 no_batch_running_stats=False, ft_emb_vec=None,
                 ncc_lambda=0.):
        
        log.info(comment)
        self.logdir_name = logdir
        self.comment = comment
        self.dataset = dataset
        self.model_name = model_name
        self.optimizer_type = optimizer_type
        self.scheduler_mode = scheduler_mode
        self.T_max = comm_rounds if T_max is None or comm_rounds > T_max else T_max
        self.save_model = save_model
        self.strategy = strategy
        self.finetuning = finetuning
        self.model_path = model_path
        self.task = task
        self.state_dict = state_dict
        self.comm_rounds = comm_rounds
        self.label_smoothing = label_smoothing
        self.fed_prox = fed_prox
        self.proximal_map = proximal_map
        self.no_batch_running_stats = no_batch_running_stats
        self.ncc_lambda = ncc_lambda
        self.time_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.trn_writer = None
        self.val_writer = None
        if get_transforms is not None:
            self.transforms = getTransformList(get_transforms, site_number, seed=1, device=self.device, var_add=(0.005, 0.05), alpha=(1.2, 2), var_mul=(0.01, 0.05), patch_size=3, swap_count=1)
        if sites is not None:
            self.trn_dls = [site['trn_dl'] for site in sites]
            self.val_dls = [site['val_dl'] for site in sites]
            self.transforms = [site['transform'] for site in sites]
            self.classes = [site['classes'] for site in sites] if sites[0]['classes'] is not None else None
            self.degs = torch.tensor([site['deg'] for site in sites], device=self.device)
            if task == 'segmentation':
                self.present_classes = [c for c_l in self.classes for c in c_l] if self.classes is not None else np.arange(18 if self.dataset == 'celeba' else 12)
            self.site_number = len(sites)
            self.site_indices = range(self.site_number) if site_indices is None else site_indices
        else:
            self.site_number = site_number
            self.trn_dls, self.val_dls = self.initDls(batch_size=batch_size, partition=partition, alpha=alpha, k_fold_val_id=k_fold_val_id, seed=seed, site_indices=site_indices)
            self.site_indices = range(site_number) if site_indices is None else site_indices
        self.iterations = iterations
        self.models = self.initModels(embed_dim=embed_dim, model_type=model_type, cifar=cifar, feature_dims=feature_dims, norm_layer=norm_layer, ft_emb_vec=ft_emb_vec)
        self.optims, self.emb_optims = self.initOptimizers(lr, finetuning, weight_decay=weight_decay, embedding_lr=embedding_lr, ffwrd_lr=ffwrd_lr)
        self.schedulers, self.emb_schedulers = self.initSchedulers(lr, embedding_lr)
        self.trn_logging = trn_logging
        assert len(self.trn_dls) == self.site_number and len(self.val_dls) == self.site_number and len(self.models) == self.site_number and len(self.optims) == self.site_number

    def initModels(self, embed_dim, model_type, cifar, feature_dims, norm_layer, ft_emb_vec):
        models, self.num_classes = get_model(self.dataset, self.model_name, self.site_number, embed_dim, model_type=model_type, task=self.task, cifar=cifar, feature_dims=feature_dims, norm_layer=norm_layer, ft_emb_vec=ft_emb_vec)

        if self.device == 'cuda':
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            for model in models:
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(self.device)
        return models

    def initOptimizers(self, lr, finetuning, weight_decay=None, embedding_lr=None, ffwrd_lr=None):
        optims = []
        emb_optims = []
        for model in self.models:
            if finetuning:
                assert self.strategy in ['finetuning', 'onlyfc', 'onlyemb', 'fffinetuning', 'fedbn', 'embbnft', 'nomerge']
                all_names = get_layer_list(task=self.task, strategy=self.strategy, model=model)
                for name, param in model.named_parameters():
                    if name not in all_names:
                        param.requires_grad = False
            else:
                all_names = list(model.state_dict().keys())
            params_to_update = []
            embedding_names = [name for name in all_names if 'embedding' in name]
            ffwrd_names = [name for name in all_names if 'generator' in name]
            if embedding_lr is not None:
                emb_params = [param for name, param in model.named_parameters() if name in embedding_names and name in all_names]
                if self.optimizer_type == 'adam':
                    emb_optim = Adam(params=emb_params, lr=embedding_lr, weight_decay=weight_decay)
                elif self.optimizer_type == 'newadam':
                    emb_optim = Adam(params=emb_params, lr=embedding_lr, weight_decay=weight_decay, betas=(0.5,0.9))
                elif self.optimizer_type == 'adamw':
                    emb_optim = AdamW(params=emb_params, lr=embedding_lr, weight_decay=weight_decay)
                elif self.optimizer_type == 'sgd':
                    emb_optim = SGD(params=emb_params, lr=embedding_lr, weight_decay=weight_decay, momentum=0.9)
                emb_optims.append(emb_optim)
            if ffwrd_lr is not None:
                params_to_update.append({'params':[param for name, param in model.named_parameters() if name in ffwrd_names and name in all_names], 'lr':ffwrd_lr})
            params_to_update.append({'params':[param for name, param in model.named_parameters() if not name in embedding_names and not name in ffwrd_names  and name in all_names]})

            if self.optimizer_type == 'adam':
                optim = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)
            elif self.optimizer_type == 'newadam':
                if self.proximal_map:
                    optim = ProxAdam(params=params_to_update, lr=lr, weight_decay=weight_decay, betas=(0.5,0.9), mu=self.fed_prox)
                else:
                    optim = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay, betas=(0.5,0.9))
            elif self.optimizer_type == 'adamw':
                optim = AdamW(params=params_to_update, lr=lr, weight_decay=weight_decay)
            elif self.optimizer_type == 'sgd':
                optim = SGD(params=params_to_update, lr=lr, weight_decay=weight_decay, momentum=0.9)

            optims.append(optim)
        return optims, emb_optims
    
    def initSchedulers(self, lr, emb_lr):
        if self.scheduler_mode is None:
            schedulers = None
        else:
            schedulers = []
            for optim in self.optims:
                if self.scheduler_mode == 'cosine':
                    scheduler = CosineAnnealingLR(optim, T_max=self.T_max, eta_min=1e-6)
                elif self.scheduler_mode == 'onecyc':
                    scheduler = OneCycleLR(optim,
                                           max_lr=lr,
                                           steps_per_epoch=self.iterations if self.iterations is not None else len(self.trn_dls[0]),
                                           epochs=self.comm_rounds)
                elif self.scheduler_mode == 'warmcos':
                    linear_warm_up = LinearLR(optim, start_factor=1e-8, total_iters=20)
                    cosine_scheduler = CosineAnnealingLR(optim, self.comm_rounds - 20, 1e-6)
                    scheduler = SequentialLR(optim, schedulers=[linear_warm_up, cosine_scheduler], milestones=[20])
                schedulers.append(scheduler)
            emb_schedulers = []
            for emb_optim in self.emb_optims:
                if self.scheduler_mode == 'cosine':
                    emb_scheduler = CosineAnnealingLR(emb_optim, T_max=self.T_max, eta_min=emb_lr * 1e-3)
                elif self.scheduler_mode == 'onecyc':
                    emb_scheduler = OneCycleLR(emb_optim,
                                           max_lr=emb_lr,
                                           steps_per_epoch=self.iterations if self.iterations is not None else len(self.trn_dls[0]),
                                           epochs=self.comm_rounds)
                elif self.scheduler_mode == 'warmcos':
                    linear_warm_up = LinearLR(emb_optim, start_factor=1e-8, total_iters=20)
                    cosine_scheduler = CosineAnnealingLR(emb_optim, self.comm_rounds - 20, emb_lr * 1e-3)
                    emb_scheduler = SequentialLR(emb_optim, schedulers=[linear_warm_up, cosine_scheduler], milestones=[20])
                emb_schedulers.append(emb_scheduler)
            
        return schedulers, emb_schedulers

    def initDls(self, batch_size, partition, alpha, k_fold_val_id, seed, site_indices):
        trn_dls, val_dls = get_dl_lists(dataset=self.dataset, partition=partition, n_site=self.site_number, batch_size=batch_size, alpha=alpha, k_fold_val_id=k_fold_val_id, seed=seed, site_indices=site_indices)
        return trn_dls, val_dls
    
    def updateGlobalEmbs(self):
        embs = self.getEmbs()
        if embs is not None:
            self.global_embs = embs.clone().detach()

    def initTensorboardWriters(self):
        tensorboard_dir = os.path.join('/home/hansel/developer/embedding/runs', self.logdir_name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=tensorboard_dir + '/trn-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=tensorboard_dir + '/val-' + self.comment)

    def train(self, state_dict=None):
        log.info("Starting {}".format(type(self).__name__))
        state_dict = self.state_dict if self.state_dict is not None else state_dict
        self.mergeModels(is_init=True, model_path=self.model_path, state_dict=state_dict)
        self.updateGlobalEmbs()

        trn_dls = self.trn_dls
        val_dls = self.val_dls

        saving_criterion = 0
        validation_cadence = 5

        if self.finetuning:
            val_metrics, imgs = self.doValidation(val_dls)
            self.logMetrics(0, 'val', val_metrics, imgs)
            metric_to_report = val_metrics['overall/accuracy'] if 'overall/accuracy' in val_metrics.keys() else val_metrics['average/mean dice']
            log.info('Round {} of {}, accuracy/dice {}, val loss {}'.format(0, self.comm_rounds, metric_to_report, val_metrics['mean loss']))

        for comm_round in range(1, self.comm_rounds + 1):
            logging_index = comm_round % 10**(math.floor(math.log(comm_round, 10))) == 0

            if comm_round == 1:
                log.info("Round {} of {}, training on {} sites, using {} device".format(
                    comm_round,
                    self.comm_rounds,
                    len(trn_dls),
                    (torch.cuda.device_count() if self.device == 'cuda' else 1),
                ))

            trn_metrics = self.doTraining(trn_dls)
            self.logMetrics(comm_round, 'trn', trn_metrics)
            self.saveEmbs(comm_round, trn_metrics)
            self.updateGlobalEmbs()

            if comm_round == 1 or comm_round % validation_cadence == 0:
                val_metrics, imgs = self.doValidation(val_dls)
                self.logMetrics(comm_round, 'val', val_metrics, imgs)
                metric_to_report = val_metrics['overall/accuracy'] if 'overall/accuracy' in val_metrics.keys() else val_metrics['average/mean dice']
                saving_criterion = max(metric_to_report, saving_criterion)

                if self.save_model and metric_to_report==saving_criterion:
                    self.saveModel(comm_round, val_metrics, trn_dls, val_dls)
                    best_state_dict = self.models[0].state_dict()
                if logging_index:
                    log.info('Round {} of {}, accuracy/dice {}, val loss {}'.format(comm_round, self.comm_rounds, metric_to_report, val_metrics['mean loss']))
            
            if self.scheduler_mode in ['cosine', 'warmcos']:
                for scheduler in self.schedulers:
                    scheduler.step()
                for emb_scheduler in self.emb_schedulers:
                    emb_scheduler.step()

            if self.site_number > 1 and not self.finetuning:
                self.mergeModels()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

        return saving_criterion, best_state_dict

    def doTraining(self, trn_dls):
        for model in self.models:
            model.train()
            if self.finetuning and self.no_batch_running_stats:
                for module in model.modules():
                    if isinstance(module, GeneralBatchNorm2d):
                        module.eval()

        metrics = []
        for ndx, trn_dl in enumerate(trn_dls):
            iter_ndx = 0
            site_metrics = self.get_empty_metrics()
            if self.iterations is not None:
                while iter_ndx < self.iterations:

                    for batch_tuple in trn_dl:
                        # with torch.autograd.detect_anomaly():
                            self.optims[ndx].zero_grad()
                            if len(self.emb_optims) > 0:
                                self.emb_optims[ndx].zero_grad()
                            loss, _ = self.computeBatchLoss(
                                batch_tuple,
                                self.models[ndx],
                                site_metrics,
                                'trn',
                                ndx)
                            loss.backward()
                            self.optims[ndx].step()
                            if len(self.emb_optims) > 0:
                                self.emb_optims[ndx].step()
                            if self.scheduler_mode == 'onecyc' and not self.finetuning:
                                for scheduler in self.schedulers:
                                    scheduler.step()
                                for emb_scheduler in self.emb_schedulers:
                                    emb_scheduler.step()
                            iter_ndx += 1
                            if iter_ndx >= self.iterations:
                                break
            else:
                for batch_tuple in trn_dl:
                    # with torch.autograd.detect_anomaly():
                        self.optims[ndx].zero_grad()
                        if len(self.emb_optims) > 0:
                            self.emb_optims[ndx].zero_grad()
                        loss, _ = self.computeBatchLoss(
                            batch_tuple,
                            self.models[ndx],
                            site_metrics,
                            'trn',
                            ndx)
                        loss.backward()
                        self.optims[ndx].step()
                        if len(self.emb_optims) > 0:
                            self.emb_optims[ndx].step()
                        if self.scheduler_mode == 'onecyc' and not self.finetuning:
                            for scheduler in self.schedulers:
                                scheduler.step()
                            for emb_scheduler in self.emb_schedulers:
                                emb_scheduler.step()

            metrics.append(site_metrics)
        trn_metrics = self.calculateGlobalMetricsFromLocal(metrics, mode='trn')

        return trn_metrics

    def doValidation(self, val_dls):
        with torch.no_grad():
            for model in self.models:
                model.eval()

            metrics = []
            need_imgs = self.task == 'segmentation'
            for ndx, val_dl in enumerate(val_dls):
                site_metrics = self.get_empty_metrics()
                for batch_tuple in val_dl:
                    # with torch.autograd.detect_anomaly():
                        _, img_list = self.computeBatchLoss(
                            batch_tuple,
                            self.models[ndx],
                            site_metrics,
                            'val',
                            ndx,
                            need_imgs
                        )
                        if img_list is not None:
                            imgs_to_save = img_list
                        need_imgs = False
                metrics.append(site_metrics)
            val_metrics = self.calculateGlobalMetricsFromLocal(metrics, mode='val')

        return val_metrics, imgs_to_save if self.task == 'segmentation' else None

    def computeBatchLoss(self, batch_tup, model, metrics, mode, site_id, need_imgs=False):
        batch, labels = batch_tup
        batch = batch.to(device=self.device, non_blocking=True).float()
        labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)
        if self.dataset in ['cifar10', 'cifar100']:
            batch = batch.permute(0, 3, 1, 2)
        if self.dataset in ['celeba']:
            batch = batch.permute(0, 3, 1, 2)
            labels = create_mask_from_onehot(labels, self.classes[site_id] if self.classes is not None else np.arange(18))

        batch, labels = transform_image(batch, labels, mode, self.transforms[site_id] if hasattr(self, 'transforms') else None, self.dataset, model=self.model_name, trn_log=self.trn_logging)

        pred = model(batch)
        pred_label = torch.argmax(pred, dim=1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        prox_term = torch.tensor(0., device=self.device)
        if self.fed_prox > 0 and not self.finetuning and not self.proximal_map:
            layers = get_layer_list(self.task, self.strategy, model=self.global_model)
            for (layer, glob_p), loc_p in zip(self.global_model.named_parameters(), model.parameters()):
                if layer in layers:
                    prox_term += torch.pow(torch.norm(glob_p - loc_p), 2)

        ncc_term = torch.tensor(0., device=self.device)
        if model.embedding is not None:
            emb_normed = (model.embedding / model.embedding.norm()).unsqueeze(0).unsqueeze(0)
            gl_emb_normed = (self.global_embs / self.global_embs.norm(dim=-1, keepdim=True)).unsqueeze(1).to(self.device)
            ncc = nn.functional.conv1d(emb_normed, gl_emb_normed).squeeze()
            ncc = torch.tanh(5*ncc - 2.5)
            # ncc = torch.stack([ncc, torch.zeros_like(ncc)])
            # ncc, _ = torch.max(ncc, 0)
            similarity = torch.ones((self.site_number), device=self.device)
            similarity[self.degs == self.degs[site_id]] *= -1
            ncc_term = (1 + similarity * ncc).sum()

        xe_loss = loss_fn(pred, labels)
        loss = xe_loss + self.fed_prox / 2 * prox_term + self.ncc_lambda * ncc_term

        metrics = self.calculateLocalMetrics(pred_label, labels, loss, metrics, mode) if metrics is not None else None

        if need_imgs:
            images = (batch[:4] - batch[:4].amin(dim=(1, 2, 3), keepdim=True)) / (batch[:4].amax(dim=(1, 2, 3), keepdim=True) - batch[:4].amin(dim=(1, 2, 3), keepdim=True))
            masks = pred_label[:4]
            img_list = [images, masks]
            return loss.sum(), img_list
        else:
            return loss.sum(), None
    
    def get_empty_metrics(self):
        metrics = {'loss':0,
                   'total':0}
        if self.task == 'classification':
            metrics['correct'] = 0
            for cls in range(self.num_classes):
                metrics[cls] = {'correct':0,
                                'total':0}
        elif self.task == 'segmentation':
            for cls in self.present_classes:
                metrics[cls] = {
                    'dice_score':0,
                    'precision':0,
                    'recall':0,
                    'true_pos':0,
                    'false_pos':0,
                    'false_neg':0,
                    'true_neg':0,
                    'total':0
                }
        return metrics
    
    def calculateLocalMetrics(self, pred_label, labels, loss, local_metrics, mode):
        # with torch.autograd.detect_anomaly():
            local_metrics['loss'] += loss.sum()
            local_metrics['total'] += pred_label.shape[0]
            if self.task == 'classification':
                if mode == 'val' or self.trn_logging:
                    correct_mask = pred_label == labels
                    correct = torch.sum(correct_mask)
                    local_metrics['correct'] += correct

                    for cls in range(self.num_classes):
                        class_mask = labels == cls
                        local_metrics[cls]['correct'] += torch.sum(correct_mask[class_mask])
                        local_metrics[cls]['total'] += torch.sum(class_mask)

            elif self.task == 'segmentation':
                eps = 1e-5

                for cls in self.present_classes:
                    cls_value = cls + 1
                    cls_label = labels == cls_value
                    cls_pred = pred_label == cls_value
                    cls_ndx_mask = torch.sum(cls_label, dim=[1, 2]) > 0

                    true_pos  = ( cls_label *  cls_pred).sum(dim=[1, 2])
                    false_pos = (~cls_label *  cls_pred).sum(dim=[1, 2])
                    false_neg = ( cls_label * ~cls_pred).sum(dim=[1, 2])
                    true_neg  = (~cls_label * ~cls_pred).sum(dim=[1, 2])

                    dice_score = (2*true_pos + eps)/(2*true_pos + false_pos + false_neg + eps) 
                    precision = (true_pos + eps)/(true_pos + false_pos + eps)
                    recall = (true_pos + eps)/(true_pos + false_neg + eps)

                    local_metrics[cls]['dice_score'] += dice_score[cls_ndx_mask].sum()
                    local_metrics[cls]['precision'] += precision[cls_ndx_mask].sum()
                    local_metrics[cls]['recall'] += recall[cls_ndx_mask].sum()
                    local_metrics[cls]['true_pos'] += true_pos[cls_ndx_mask].sum()
                    local_metrics[cls]['false_pos'] += false_pos[cls_ndx_mask].sum()
                    local_metrics[cls]['false_neg'] += false_neg[cls_ndx_mask].sum()
                    local_metrics[cls]['true_neg'] += true_neg[cls_ndx_mask].sum()
                    local_metrics[cls]['total'] += cls_ndx_mask.sum()

    def calculateGlobalMetricsFromLocal(self, local_metrics, mode):
        # with torch.autograd.detect_anomaly():
            eps = 1e-5

            total = sum([d['total'] for d in local_metrics])
            gl_metrics = {'mean loss':sum([d['loss'] for d in local_metrics]) / total}

            if self.task == 'classification':
                if mode == 'val' or self.trn_logging:
                    gl_metrics['overall/accuracy'] = sum([d['correct'] for d in local_metrics]) / total

                    for site_ndx, local_m in enumerate(local_metrics):
                        gl_metrics['accuracy_by_site/{}'.format(site_ndx)] = local_m['correct'] / local_m['total']
                        gl_metrics['loss by site/{}'.format(site_ndx)] = local_m['loss'] / local_m['total']

                    for cls in range(self.num_classes):
                        cls_total = sum([d[cls]['total'] for d in local_metrics])
                        gl_metrics['accuracy by class/{}'.format(cls)] = sum([d[cls]['correct'] for d in local_metrics]) / cls_total

            elif self.task == 'segmentation':
                for cls in self.present_classes:
                    cls_total = sum([d[cls]['total'] for d in local_metrics])
                    tp = sum([d[cls]['true_pos'] for d in local_metrics])
                    fp = sum([d[cls]['false_pos'] for d in local_metrics])
                    fn = sum([d[cls]['false_neg'] for d in local_metrics])
                    
                    gl_metrics['overall dice per class/{}'.format(cls)] = (2*tp + eps) / (2*tp + fp + fn + eps) if cls_total > 0 else torch.tensor(math.nan)
                    gl_metrics['overall precision per class/{}'.format(cls)] = (tp + eps) / (tp + fp + eps) if cls_total > 0 else torch.tensor(math.nan)
                    gl_metrics['overall recall per class/{}'.format(cls)] = (tp + eps) / (tp + fn + eps) if cls_total > 0 else torch.tensor(math.nan)

                    gl_metrics['avg dice per class/{}'.format(cls)] = sum([d[cls]['dice_score'] for d in local_metrics]) / cls_total if cls_total > 0 else torch.tensor(math.nan)
                    gl_metrics['avg precision per class/{}'.format(cls)] = sum([d[cls]['precision'] for d in local_metrics]) / cls_total if cls_total > 0 else torch.tensor(math.nan)
                    gl_metrics['avg recall per class/{}'.format(cls)] = sum([d[cls]['recall'] for d in local_metrics]) / cls_total if cls_total > 0 else torch.tensor(math.nan)

                gl_metrics['overall/mean dice'] = torch.nanmean(torch.tensor([gl_metrics['overall dice per class/{}'.format(cls)] for cls in self.present_classes]))
                gl_metrics['overall/mean precision'] = torch.nanmean(torch.tensor([gl_metrics['overall precision per class/{}'.format(cls)] for cls in self.present_classes]))
                gl_metrics['overall/mean recall'] = torch.nanmean(torch.tensor([gl_metrics['overall recall per class/{}'.format(cls)] for cls in self.present_classes]))

                gl_metrics['average/mean dice'] = torch.nanmean(torch.tensor([gl_metrics['avg dice per class/{}'.format(cls)] for cls in self.present_classes]))
                gl_metrics['average/mean precision'] = torch.nanmean(torch.tensor([gl_metrics['avg precision per class/{}'.format(cls)] for cls in self.present_classes]))
                gl_metrics['average/mean recall'] = torch.nanmean(torch.tensor([gl_metrics['avg recall per class/{}'.format(cls)] for cls in self.present_classes]))
            
            for k in gl_metrics.keys():
                gl_metrics[k] = gl_metrics[k].detach().cpu()

            return gl_metrics


    def logMetrics(
        self,
        comm_round,
        mode_str,
        metrics,
        imgs=None
    ):
        iter_number = comm_round * self.iterations if self.iterations is not None else comm_round * len(self.trn_dls[0])
        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')
        for k, v in metrics.items():
            writer.add_scalar(k, scalar_value=v, global_step=iter_number)
        if imgs is not None:
            img_grid = make_grid(imgs[0])
            mask_grid = make_grid(imgs[1].unsqueeze(1).float(), normalize=True)
            writer.add_image('images/image', img_grid, global_step=iter_number, dataformats='CHW')
            writer.add_image('images/mask', mask_grid, global_step=iter_number, dataformats='CHW')
        writer.flush()

    def saveModel(self, epoch_ndx, val_metrics, trn_dls, val_dls):
        model_file_path = os.path.join('/home/hansel/developer/embedding/saved_models',
                                       self.logdir_name,
                                       f'{self.time_str}-{self.comment}.state')
        data_file_path = os.path.join('/home/hansel/developer/embedding/saved_metrics',
                                      self.logdir_name,
                                      f'{self.time_str}-{self.comment}.state')
        
        os.makedirs(os.path.dirname(model_file_path), mode=0o755, exist_ok=True)
        os.makedirs(os.path.dirname(data_file_path), mode=0o755, exist_ok=True)

        data_state = {'valmetrics':val_metrics,
                      'epoch': epoch_ndx,}
        model_state = {'epoch': epoch_ndx}
        model_state['model_state'] = self.models[0].state_dict()
        layer_list = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
        for ndx, model in enumerate(self.models):
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            state_dict = model.state_dict()
            if self.finetuning:
                site_state_dict = {key: state_dict[key] for key in state_dict.keys() if key in layer_list}
            else:
                site_state_dict = {key: state_dict[key] for key in state_dict.keys() if key not in layer_list}
            model_state[ndx] = {
                'site_model_state': site_state_dict,
                'optimizer_state': self.optims[ndx].state_dict(),
                'scheduler_state': self.schedulers[ndx].state_dict(),
                }
            if len(self.emb_optims) > 0:
                model_state[ndx]['emb_optimizer_state'] = self.emb_optims[ndx].state_dict()
                model_state[ndx]['emb_scheduler_state'] = self.emb_schedulers[ndx].state_dict()
            if hasattr(model, 'embedding') and model.embedding is not None:
                data_state[ndx] = {'emb_vector':model.embedding.detach().cpu()}

        torch.save(model_state, model_file_path)
        log.debug("Saved model params to {}".format(model_file_path))
        torch.save(data_state, data_file_path)
        log.debug("Saved training metrics to {}".format(data_file_path))

    def getEmbs(self):
        if hasattr(self.models[0], 'embedding') and self.models[0].embedding is not None:
            embeddings = torch.zeros((len(self.models), self.models[0].embedding.shape[0]))
            for i, model in enumerate(self.models):
                embeddings[i] = model.embedding
        else:
            embeddings = None
        return embeddings

    
    def saveEmbs(self, comm_round, metrics):
        embedding_file_path = os.path.join('/home/hansel/developer/embedding/embeddings',
                                      self.logdir_name,
                                      f'{self.time_str}-{self.comment}',
                                      f'{comm_round}.pt')
        os.makedirs(os.path.dirname(os.path.dirname(embedding_file_path)), mode=0o755, exist_ok=True)
        os.makedirs(os.path.dirname(embedding_file_path), mode=0o755, exist_ok=True)
        embeddings = self.getEmbs()
        if embeddings is not None:
            dict_to_save = {'embedding':embeddings,
                            'metrics':metrics}
            torch.save(dict_to_save, embedding_file_path)
            log.debug("Saved embeddings to {}".format(embedding_file_path))

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
            state_dict.pop('embedding', None)
            for model in self.models:
                model.load_state_dict(state_dict, strict=False)
            self.global_model = copy.deepcopy(self.models[0])
            layer_list = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
            for model in self.models:
                for n, p in model.named_parameters():
                    if n in layer_list and self.proximal_map:
                        p.global_weight = state_dict[n]
        else:
            layer_list = get_layer_list(task=self.task, strategy=self.strategy, model=self.models[0])
            state_dicts = [model.state_dict() for model in self.models]
            updated_params = {layer: torch.zeros_like(state_dicts[0][layer]) for layer in layer_list}

            site_weights = np.array([len(dl.dataset) for dl in self.trn_dls])
            for layer in layer_list:
                for weight, state_dict in zip(site_weights, state_dicts):
                    updated_params[layer] += weight * state_dict[layer]
                updated_params[layer] = updated_params[layer] / site_weights.sum()

            self.global_model.load_state_dict(updated_params, strict=False)
            for model in self.models:
                model.load_state_dict(updated_params, strict=False)
                for n, p in model.named_parameters():
                    if n in layer_list and self.proximal_map:
                        p.global_weight = updated_params[n]


if __name__ == '__main__':
    EmbeddingTraining().train()
