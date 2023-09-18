import os
import datetime
import math

import torch
from torchvision.transforms import Resize
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, LBFGS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture as SkGMM
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

class LayerPersonalisationTrainingApp:
    def __init__(self, epochs=500, batch_size=128, logdir='test', lr=1e-3,
                 comment='dwlpt', dataset='cifar10', site_number=1,
                 model_name='resnet18emb', optimizer_type='newadam',
                 scheduler_mode='cosine', pretrained=False, T_max=500,
                 label_smoothing=0.0, save_model=False, partition='regular',
                 alpha=None, strategy='noembed', finetuning=False, embed_dim=2,
                 model_path=None, embedding_lr=None, ffwrd_lr=None, gmm_components=None,
                 single_vector_update=False, vector_update_batch=1000, vector_update_lr=1,
                 layer_number=4, gmm_reg=False, k_fold_val_id=None, seed=None,
                 site_indices=None, input_perturbation=False, use_hdf5=False):

        log.info(locals())
        self.epochs = epochs
        self.logdir_name = logdir
        self.comment = comment
        self.dataset = dataset
        self.site_number = site_number
        self.model_name = model_name
        self.optimizer_type = optimizer_type
        self.scheduler_mode = scheduler_mode
        self.pretrained = pretrained
        if epochs > T_max:
            self.T_max = epochs
        else:
            self.T_max = T_max
        self.label_smoothing = label_smoothing
        self.save_model = save_model
        self.strategy = strategy
        self.finetuning = finetuning
        self.embed_dim = embed_dim
        self.model_path = model_path
        self.gmm_components = gmm_components
        self.single_vector_update = single_vector_update
        self.vector_update_batch = vector_update_batch
        self.vector_update_lr = vector_update_lr
        self.input_perturbation = input_perturbation
        if site_indices is None:
            site_indices = range(site_number)
        self.site_indices = site_indices
        self.use_hdf5 = use_hdf5
        self.time_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.logdir_name)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.trn_dls, self.val_dls = self.initDls(batch_size=batch_size, partition=partition, alpha=alpha, k_fold_val_id=k_fold_val_id, seed=seed, site_indices=site_indices)
        self.site_number = len(site_indices)
        self.models = self.initModels(embed_dim=embed_dim, layer_number=layer_number)
        self.mergeModels(is_init=True, model_path=model_path)
        self.optims = self.initOptimizers(lr, finetuning, embedding_lr=embedding_lr, ffwrd_lr=ffwrd_lr)
        self.schedulers = self.initSchedulers()
        if gmm_components is not None:
            self.trn_gmms, self.val_gmms = self.initGMMs(gmm_reg=gmm_reg)
            self.trn_vector_model, self.val_vector_model, self.trn_vector_optim, self.val_vector_optim = self.initEmbeddingVector(layer_number=layer_number)
        assert len(self.trn_dls) == self.site_number and len(self.val_dls) == self.site_number and len(self.models) == self.site_number and len(self.optims) == self.site_number

    def initModels(self, embed_dim, layer_number):
        models, self.num_classes = get_model(self.dataset, self.model_name, self.site_number, embed_dim, layer_number, self.pretrained)

        if 'embedding.weight' in '\t'.join(models[0].state_dict().keys()):
            if embed_dim > 2:
                self.mu_init = np.eye(self.site_number, embed_dim)
            else:
                mu_init = np.exp((2 * np.pi * 1j/ self.site_number)*np.arange(0,self.site_number))
                self.mu_init = np.stack([np.real(mu_init), np.imag(mu_init)], axis=1)
            for i, model in enumerate(models):
                init_weight = torch.from_numpy(self.mu_init[i]).repeat(self.site_number).reshape(self.site_number, embed_dim)
                model.embedding.weight = nn.Parameter(init_weight)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            for i, model in enumerate(models):
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(self.device)
        return models

    def initOptimizers(self, lr, finetuning, embedding_lr=None, ffwrd_lr=None):
        optims = []
        weight_decay = 0.
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
                    ffwrd_names = [name for name in all_names if 'ffwrd' in name]
                    params_to_update.append({'params':[param for name, param in model.named_parameters() if name in ffwrd_names], 'lr':ffwrd_lr})
                params_to_update.append({'params':[param for name, param in model.named_parameters() if not name in embedding_names and not name in ffwrd_names]})

            if self.optimizer_type == 'adam':
                optim = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay)
            elif self.optimizer_type == 'newadam':
                optim = Adam(params=params_to_update, lr=lr, weight_decay=weight_decay, betas=(0.5,0.9))
            elif self.optimizer_type == 'adamw':
                optim = AdamW(params=params_to_update, lr=lr, weight_decay=0.05)
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
            
    def initGMMs(self, gmm_reg=False):
        trn_gmms = []
        val_gmms = []
        for i in range(self.site_number):
            trn_gmms.append(SkGMM(n_components=self.gmm_components, warm_start=True, means_init=[self.mu_init[i]]*self.gmm_components))
            val_gmms.append(SkGMM(n_components=self.gmm_components, warm_start=True, means_init=[self.mu_init[i]]*self.gmm_components))
        return trn_gmms, val_gmms
            
    def initEmbeddingVector(self, layer_number):
        if self.dataset == 'mnist':
            trn_size = 60000
            val_size = 10000
        elif self.dataset == 'cifar10':
            trn_size = 50000
            val_size = 10000
        elif self.dataset == 'cifar100':
            trn_size = 50000
            val_size = 10000
        elif self.dataset == 'imagenet':
            trn_size = 100000
            val_size = 10000

        trn_idx_map, val_idx_map = {}, {}
        for i in range(self.site_number):
            trn_idx_map[i] = self.trn_dls[i].dataset.indices
            val_idx_map[i] = self.val_dls[i].dataset.indices
        
        trn_dls_emb_vector, val_dls_emb_vector = get_dl_lists(dataset=self.dataset, batch_size=self.vector_update_batch, partition='given', n_site=self.site_number, net_dataidx_map_train=trn_idx_map, net_dataidx_map_test=val_idx_map, shuffle=False, use_hdf5=self.use_hdf5)
        self.vector_trn_dls, self.vector_val_dls = trn_dls_emb_vector, val_dls_emb_vector

        trn_init_vectors = torch.empty(trn_size, self.embed_dim, dtype=torch.float)
        val_init_vectors = torch.empty(val_size, self.embed_dim, dtype=torch.float)
        for i in range(self.site_number):
            cov_matrix = np.eye(self.embed_dim)*0.1
            v = torch.from_numpy(np.random.multivariate_normal(self.mu_init[i], cov_matrix, size=len(trn_idx_map[i])))
            trn_init_vectors[trn_idx_map[i]] = v.to(dtype=torch.float)
            v = torch.from_numpy(np.random.multivariate_normal(self.mu_init[i], cov_matrix, size=len(val_idx_map[i])))
            val_init_vectors[val_idx_map[i]] = v.to(dtype=torch.float)

        trn_vector_model = self.initModels(self.embed_dim, layer_number=layer_number)[0]
        if self.single_vector_update:
            trn_vector_model.embedding = nn.Embedding(int(trn_size/self.vector_update_batch), embedding_dim=self.embed_dim, device=self.device)
        else:
            trn_vector_model.embedding = nn.Embedding(trn_size, embedding_dim=self.embed_dim, device=self.device)
            trn_vector_model.embedding.weight = nn.Parameter(trn_init_vectors.to(device=self.device))
        for name, param in trn_vector_model.named_parameters():
            if 'embedding.weight' not in name:
                param.requires_grad = False

        trn_vector_optim = Adam([trn_vector_model.embedding.weight], lr=self.vector_update_lr, betas=(.5,.9))
        
        val_vector_model = self.initModels(self.embed_dim, layer_number=layer_number)[0]
        if self.single_vector_update:
            val_vector_model.embedding = nn.Embedding(int(val_size/self.vector_update_batch), embedding_dim=self.embed_dim, device=self.device)
        else:
            val_vector_model.embedding = nn.Embedding(val_size, embedding_dim=self.embed_dim, device=self.device)
            val_vector_model.embedding.weight = nn.Parameter(val_init_vectors.to(device=self.device))
        for name, param in val_vector_model.named_parameters():
            if 'embedding.weight' not in name:
                param.requires_grad = False

        val_vector_optim = Adam([val_vector_model.embedding.weight], lr=self.vector_update_lr, betas=(.5,.9))

        return trn_vector_model, val_vector_model, trn_vector_optim, val_vector_optim

    def main(self):
        log.info("Starting {}".format(type(self).__name__))

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

                if epoch_ndx < 101 or epoch_ndx % 100 == 0:
                    log.info('Epoch {} of {}, accuracy/miou {}, val loss {}'.format(epoch_ndx, self.epochs, accuracy, loss))

            if self.gmm_components is not None:
                self.updateEmbeddingVectors()
                self.fitGMM(epoch_ndx)
            
            
            if self.scheduler_mode == 'cosine' and not self.finetuning:
                for scheduler in self.schedulers:
                    scheduler.step()
                    # log.debug(self.scheduler.get_last_lr())

            if self.site_number > 1 and not self.finetuning:
                self.mergeModels()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

        return saving_criterion

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
                # with torch.autograd.detect_anomaly():
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
                    if self.optimizer_type == 'lbfgs':
                        self.optims[ndx].step(closure)
                    else:
                        loss = closure()
                        self.optims[ndx].step()

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
            batch = perturb(batch, self.site_indices[site_id])
        if mode == 'trn':
            batch = aug_image(batch, self.dataset)
        if self.model_name[:6] == 'maxvit':
            resize = Resize(224, antialias=True)
            batch = resize(batch)

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

        return loss.mean(), accuracy
    
    def updateEmbeddingVectors(self):
        self.copyModelToVectorModel()

        for i in range(self.site_number):
            for batch_ndx, batch_tup in enumerate(self.vector_trn_dls[i]):
                self.trn_vector_optim.zero_grad()
                imgs, labels, img_ids = batch_tup
                imgs = imgs.to(device=self.device).float().permute(0, 3, 1, 2)
                if self.model_name[:9] == 'maxvitemb':
                    resize = Resize(224, antialias=True)
                    imgs = resize(imgs)
                labels = labels.to(device=self.device).to(dtype=torch.long)
                img_ids = img_ids.to(device=self.device).to(dtype=torch.long)
                preds = torch.zeros(imgs.shape[0], 10, device=self.device)
                if self.single_vector_update:
                    preds = self.trn_vector_model(imgs, torch.tensor(batch_ndx, device=self.device))
                else:
                    preds = self.trn_vector_model(imgs, img_ids)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(preds, labels)
                loss.backward()
                self.trn_vector_optim.step()
            log.debug('done with trn-site {}'.format(i))

        for i in range(self.site_number):
            for batch_ndx, batch_tup in enumerate(self.vector_val_dls[i]):
                self.val_vector_optim.zero_grad()
                imgs, labels, img_ids = batch_tup
                imgs = imgs.to(device=self.device).float().permute(0, 3, 1, 2)
                if self.model_name[:9] == 'maxvitemb':
                    resize = Resize(224, antialias=True)
                    imgs = resize(imgs)
                labels = labels.to(device=self.device).to(dtype=torch.long)
                img_ids = img_ids.to(device=self.device).to(dtype=torch.long)
                preds = torch.zeros(imgs.shape[0], 10, device=self.device)
                if self.single_vector_update:
                    preds = self.val_vector_model(imgs, torch.tensor(batch_ndx, device=self.device))
                else:
                    preds = self.val_vector_model(imgs, img_ids)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(preds, labels)
                loss.backward()
                self.val_vector_optim.step()
            log.debug('done with val-site {}'.format(i))
        log.debug('completed vector update')
    
    def fitGMM(self, epoch_ndx):
        trn_vectors, val_vectors = self.extractEmbeddingVectors()
        trn_vectors = trn_vectors.to(device='cpu')
        val_vectors = val_vectors.to(device='cpu')
        trn_img_indices, val_img_indices = self.getImageIndicesBySite()
        save_path = os.path.join(
            'gmm_data',
            self.logdir_name,
            '{}-{}.pt'.format(self.time_str, self.comment))
        
        if epoch_ndx == 1:
            state = {}
        else:
            state = torch.load(save_path)
        state[epoch_ndx] = {'trn':{}, 'val':{}}
        for i, indices in enumerate(trn_img_indices):
            self.trn_gmms[i].fit(trn_vectors[indices])
            mu = self.trn_gmms[i].means_
            var = self.trn_gmms[i].covariances_
            pred = self.trn_gmms[i].predict(trn_vectors[indices])
            state[epoch_ndx]['trn'][i] = {'vectors':trn_vectors[indices],
                               'pred':torch.tensor(pred, device='cpu'),
                               'mu':torch.tensor(mu, device='cpu'),
                               'var':torch.tensor(var, device='cpu')}
        for i, indices in enumerate(val_img_indices):
            self.val_gmms[i].fit(val_vectors[indices])
            mu = self.val_gmms[i].means_
            var = self.val_gmms[i].covariances_
            pred = self.val_gmms[i].predict(val_vectors[indices])
            state[epoch_ndx]['val'][i] = {'vectors':val_vectors[indices],
                               'pred':torch.tensor(pred, device='cpu'),
                               'mu':torch.tensor(mu, device='cpu'),
                               'var':torch.tensor(var, device='cpu')}
        os.makedirs(os.path.dirname(save_path), mode=0o755, exist_ok=True)
        torch.save(state, save_path)
        log.debug('fit gmm and saved data')

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
        file_path = os.path.join(
            'saved_models',
            self.logdir_name,
            '{}-{}.state'.format(
                self.time_str,
                self.comment
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        state = {'valmetrics':val_metrics,
                 'epoch': epoch_ndx}
        for ndx, model in enumerate(self.models):
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            state[ndx] = {
                'model_state': model.state_dict(),
                'model_name': type(model).__name__,
                'optimizer_state': self.optims[ndx].state_dict(),
                'optimizer_name': type(self.optims[ndx]).__name__,
                'trn_labels': trn_dls[ndx].dataset.labels,
                'val_labels': val_dls[ndx].dataset.labels,
                'trn_indices': trn_dls[ndx].dataset.indices,
                'val_indices': val_dls[ndx].dataset.indices,
                }
            if 'embedding.weight' in '\t'.join(model.state_dict().keys()):
                state[ndx]['emb_vector'] = model.state_dict()['embedding.weight'][ndx]

        torch.save(state, file_path)
        log.debug("Saved model params to {}".format(file_path))

    def mergeModels(self, is_init=False, model_path=None):
        if is_init:
            if model_path is not None:
                loaded_dict = torch.load(model_path)
                if 'model_state' in loaded_dict.keys():
                    state_dict = loaded_dict['model_state']
                else:
                    state_dict = loaded_dict[0]['model_state']
            else:
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

    def copyModelToVectorModel(self):
        model_state_dict = self.models[0].state_dict()
        del model_state_dict['embedding.weight']
        self.trn_vector_model.load_state_dict(model_state_dict, strict=False)
        self.val_vector_model.load_state_dict(model_state_dict, strict=False)
        for name, param in self.trn_vector_model.named_parameters():
            if name != 'embedding.weight':
                param.requires_grad = False
        for name, param in self.val_vector_model.named_parameters():
            if name != 'embedding.weight':
                param.requires_grad = False

    def extractEmbeddingVectors(self):
        trn_state_dict = self.trn_vector_model.state_dict()
        val_state_dict = self.val_vector_model.state_dict()
        return trn_state_dict['embedding.weight'], val_state_dict['embedding.weight']
    
    def getImageIndicesBySite(self):
        trn_img_indices = []
        val_img_indices = []
        for trn_dl in self.trn_dls:
            trn_img_indices.append(trn_dl.dataset.indices)
        for val_dl in self.val_dls:
            val_img_indices.append(val_dl.dataset.indices)
        return trn_img_indices, val_img_indices


if __name__ == '__main__':
    LayerPersonalisationTrainingApp().main()
