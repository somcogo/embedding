import sys
import argparse
import os
import datetime
import shutil
import random

import torch
from torch import autograd
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, LBFGS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from models.model import ResNetWithEmbeddings, ResNet34Model, ResNet18Model
from utils.logconf import logging
from utils.data_loader import get_cifar10_dl, get_cifar100_dl, get_dl_lists
from utils.ops import aug_image
from utils.merge_strategies import get_layer_list

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class LayerPersonalisationTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, comment=None, dataset='cifar10', site_number=None, model_name=None, optimizer_type=None, scheduler_mode=None, label_smoothing=None, T_max=None, pretrained=None, aug_mode=None, save_model=None, partition=None, alpha=None, strategy=None, model_path=None, finetuning=False, embed_dim=None, also_last_layer=None, embedding_lr=None, ffwrd_lr=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser(description="Test training")
        parser.add_argument("--epochs", default=2, type=int, help="number of training epochs")
        parser.add_argument("--batch_size", default=64, type=int, help="number of batch size")
        parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--in_channels", default=3, type=int, help="number of image channels")
        parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
        parser.add_argument("--dataset", default='cifar10', type=str, help="dataset to train on")
        parser.add_argument("--site_number", default=1, type=int, help="number of sites")
        parser.add_argument("--model_name", default='resnet34emb', type=str, help="name of the model to use")
        parser.add_argument("--optimizer_type", default='adam', type=str, help="type of optimizer to use")
        parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing in Cross Entropy Loss")
        parser.add_argument("--T_max", default=1000, type=int, help="T_max in Cosine LR scheduler")
        parser.add_argument("--pretrained", default=False, type=bool, help="use pretrained model")
        parser.add_argument("--aug_mode", default='segmentation', type=str, help="mode of data augmentation")
        parser.add_argument("--scheduler_mode", default=None, type=str, help="choice of LR scheduler")
        parser.add_argument("--save_model", default=False, type=bool, help="save models during training")
        parser.add_argument("--partition", default='regular', type=str, help="how to partition the data among sites")
        parser.add_argument("--alpha", default=None, type=float, help="alpha used for the Dirichlet distribution")
        parser.add_argument("--strategy", default='all', type=str, help="merging strategy")
        parser.add_argument("--finetuning", default=False, type=bool, help="true if we are finetuning the embeddings on a new site")
        parser.add_argument("--embed_dim", default=2, type=int, help="dimension of the latent space where the site number is mapped")
        parser.add_argument("--also_last_layer", default=False, type=bool, help="whether to also finetune the last layer")
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')

        self.args = parser.parse_args()
        # self.epochs = epochs if epochs is not None else 2
        # self.batch_size = batch_size if batch_size is not None else 128
        # self.logdir = logdir if logdir is not None else 'test'
        # self.comment = comment if comment is not None else 'dwlpt'
        # self.dataset = dataset if dataset is not None else 'cifar10'
        # self.site_number = site_number if site_number is not None else 1
        # self.model_name = model_name if model_name is not None else 'resnet18emb'
        # self.optimizer_type = optimizer_type if optimizer_type is not None else 'sgd'
        # self.scheduler_mode = scheduler_mode 
        # self.pretrained = pretrained if pretrained is not None else 500
        # self.T_max = T_max if T_max is not None else 500
        # self.T_max = T_max if T_max is not None else 500
        # self.T_max = T_max if T_max is not None else 500
        # self.T_max = T_max if T_max is not None else 500
        self.model_path = model_path
        if epochs is not None:
            self.args.epochs = epochs
        if batch_size is not None:
            self.args.batch_size = batch_size
        if logdir is not None:
            self.args.logdir = logdir
        if lr is not None:
            self.args.lr = lr
        if comment is not None:
            self.args.comment = comment
        if dataset is not None:
            self.args.dataset = dataset
        if site_number is not None:
            self.args.site_number = site_number
        if model_name is not None:
            self.args.model_name = model_name
        if optimizer_type is not None:
            self.args.optimizer_type = optimizer_type
        if label_smoothing is not None:
            self.args.label_smoothing = label_smoothing
        if T_max is not None:
            self.args.T_max = T_max
        if pretrained is not None:
            self.args.pretrained = pretrained
        if aug_mode is not None:
            self.args.aug_mode = aug_mode
        if scheduler_mode is not None:
            self.args.scheduler_mode = scheduler_mode
        if save_model is not None:
            self.args.save_model = save_model
        if partition is not None:
            self.args.partition = partition
        if alpha is not None:
            self.args.alpha = alpha
        if strategy is not None:
            self.args.strategy = strategy
        if finetuning is not None:
            self.args.finetuning = finetuning
        if embed_dim is not None:
            self.args.embed_dim = embed_dim
        if also_last_layer is not None:
            self.args.also_last_layer = also_last_layer
        self.time_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.models = self.initModels()
        self.mergeModels(is_init=True, model_path=model_path)
        self.optims = self.initOptimizers(finetuning, also_last_layer, embedding_lr=embedding_lr, ffwrd_lr=ffwrd_lr)
        self.schedulers = self.initSchedulers()
        trn_dls, val_dls = self.initDls()
        self.trn_dls = trn_dls
        self.val_dls = val_dls

    def initModels(self):
        if self.args.dataset == 'cifar10':
            num_classes = 10
            in_channels = 3
        elif self.args.dataset == 'cifar100':
            num_classes = 100
            in_channels = 3
        elif self.args.dataset == 'pascalvoc':
            num_classes = 21
            in_channels = 3
        elif self.args.dataset == 'mnist':
            num_classes = 10
            in_channels = 1
        self.num_classes = num_classes
        models = []
        for _ in range(self.args.site_number):
            if self.args.model_name == 'resnet34emb':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[3, 4, 6, 3], site_number=self.args.site_number, embed_dim=self.args.embed_dim)
            elif self.args.model_name == 'resnet18emb':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim)
            elif self.args.model_name == 'resnet18embhypnn1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=1)
            elif self.args.model_name == 'resnet18embhypnn2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=2)
            elif self.args.model_name == 'resnet18lightweight1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=1, lightweight=True)
            elif self.args.model_name == 'resnet18lightweight2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=2, lightweight=True)
            elif self.args.model_name == 'resnet18affine1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True)
            elif self.args.model_name == 'resnet18affine2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True)
            elif self.args.model_name == 'resnet18medium1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True, medium_ffwrd=True)
            elif self.args.model_name == 'resnet18medium2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.args.site_number, embed_dim=self.args.embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True, medium_ffwrd=True)
            elif self.args.model_name == 'resnet34':
                model = ResNet34Model(num_classes=num_classes, in_channels=in_channels, pretrained=self.args.pretrained)
            elif self.args.model_name == 'resnet18':
                model = ResNet18Model(num_classes=num_classes, in_channels=in_channels, pretrained=self.args.pretrained)
            models.append(model)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            for model in models:
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                model = model.to(self.device)
        return models

    def initOptimizers(self, finetuning, also_last_layer, embedding_lr=None, ffwrd_lr=None):
        optims = []
        if self.args.model_name == 'resnet18embhypnn1' or self.args.model_name == 'resnet18embhypnn2' or self.args.model_name == 'resnet18lightweight1' or self.args.model_name == 'resnet18lightweight2':
            weight_decay = 1
        else:
            weight_decay = 0.0001
        for model in self.models:
            if finetuning:
                if self.args.model_name == 'resnet18emb' and also_last_layer:
                    layer_list = get_layer_list(self.args.model_name, strategy='finetuning2')
                else:
                    layer_list = get_layer_list(self.args.model_name, strategy='finetuning')
                params_to_update = []
                for name, param in model.named_parameters():
                    if name in layer_list:
                        params_to_update.append(param)
                    else:
                        param.requires_grad = False
            else:
                all_names = [name for name, _ in model.named_parameters()]
                embedding_names = [name for name in all_names if name.split('.')[0] == 'embedding'] if embedding_lr is not None else []
                ffwrd_names = [name for name in all_names if 'ffwrd' in name] if ffwrd_lr is not None else []
                params_to_update = []
                params_to_update.append({'params':[param for name, param in model.named_parameters() if name in embedding_names], 'lr':embedding_lr})
                params_to_update.append({'params':[param for name, param in model.named_parameters() if name in ffwrd_names], 'lr':ffwrd_lr})
                params_to_update.append({'params':[param for name, param in model.named_parameters() if not name in embedding_names and not name in ffwrd_names]})

            if self.args.optimizer_type == 'adam':
                optim = Adam(params=params_to_update, lr=self.args.lr, weight_decay=weight_decay)
            elif self.args.optimizer_type == 'adamw':
                optim = AdamW(params=params_to_update, lr=self.args.lr, weight_decay=0.05)
            elif self.args.optimizer_type == 'sgd':
                optim = SGD(params=params_to_update, lr=self.args.lr, weight_decay=weight_decay, momentum=0.9)
            elif self.args.optimizer_type == 'lbfgs':
                optim = LBFGS(params=params_to_update, lr=self.args.lr, line_search_fn='strong_wolfe')
            optims.append(optim)
        return optims
    
    def initSchedulers(self):
        if self.args.scheduler_mode is None:
            schedulers = None
        else:
            schedulers = []
            for optim in self.optims:
                if self.args.scheduler_mode == 'cosine':
                    scheduler = CosineAnnealingLR(optim, T_max=self.args.T_max)
                schedulers.append(scheduler)
            
        return schedulers

    def initDls(self):
        index_dict = torch.load('models/saved_index_maps') if self.args.partition == 'given' else None
        trn_idx_map = index_dict[self.args.site_number][self.args.alpha]['trn'] if index_dict is not None else None
        val_idx_map = index_dict[self.args.site_number][self.args.alpha]['val'] if index_dict is not None else None
        trn_dls, val_dls = get_dl_lists(dataset=self.args.dataset, partition=self.args.partition, n_site=self.args.site_number, batch_size=self.args.batch_size, alpha=self.args.alpha, net_dataidx_map_train=trn_idx_map, net_dataidx_map_test=val_idx_map)
        return trn_dls, val_dls

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        trn_dls = self.trn_dls
        val_dls = self.val_dls

        saving_criterion = 0
        validation_cadence = 5

        if self.args.finetuning:
            val_metrics, accuracy, loss = self.doValidation(0, val_dls)
            self.logMetrics(0, 'val', val_metrics)
            log.info('Epoch {} of {}, accuracy/miou {}, val loss {}'.format(0, self.args.epochs, accuracy, loss))

        for epoch_ndx in range(1, self.args.epochs + 1):

            if epoch_ndx == 1:
                log.info("Epoch {} of {}, training on {} sites, using {} device".format(
                    epoch_ndx,
                    self.args.epochs,
                    len(trn_dls),
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trn_metrics = self.doTraining(epoch_ndx, trn_dls)
            self.logMetrics(epoch_ndx, 'trn', trn_metrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                val_metrics, accuracy, loss = self.doValidation(epoch_ndx, val_dls)
                self.logMetrics(epoch_ndx, 'val', val_metrics)
                saving_criterion = max(accuracy, saving_criterion)

                if self.args.save_model and (accuracy==saving_criterion or self.args.finetuning):
                    self.saveModel(epoch_ndx, val_metrics, trn_dls, val_dls)

                log.info('Epoch {} of {}, accuracy/miou {}, val loss {}'.format(epoch_ndx, self.args.epochs, accuracy, loss))
            
            if self.args.scheduler_mode == 'cosine':
                for scheduler in self.schedulers:
                    scheduler.step()
                    # log.debug(self.scheduler.get_last_lr())

            if self.args.site_number > 1 and not self.args.finetuning:
                self.mergeModels()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, trn_dls):
        for model in self.models:
            model.train()

        trn_metrics = torch.zeros(2 + 2*self.args.site_number + self.num_classes, device=self.device)
        loss = 0
        correct = 0
        total = 0
        correct_by_class = torch.zeros(self.num_classes, device=self.device)
        total_by_class = torch.zeros(self.num_classes, device=self.device)
        for ndx, trn_dl in enumerate(trn_dls):
            local_trn_metrics = torch.zeros(2 + 2*self.num_classes, len(trn_dl), device=self.device)

            for batch_ndx, batch_tuple in enumerate(trn_dl):
                # try:
                #     with torch.autograd.detect_anomaly():
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
                        if self.args.optimizer_type == 'lbfgs':
                            self.optims[ndx].step(closure)
                        else:
                            loss = closure()
                            self.optims[ndx].step()
                # except:
                #     print('epoch', epoch_ndx, 'batch', batch_ndx)
                    # for name, param in self.models[ndx].named_parameters():
                    #     if param.grad is None:
                    #         print(name, 'None')
                    #     elif param.grad.norm()>1000:
                    #         print(name, param.grad.norm())
                #     raise
                # finally:
                #     print('epoch', epoch_ndx, 'batch', batch_ndx)
                #     for name, param in self.models[ndx].named_parameters():
                #         if param is None:
                #             print(name, 'None')
                #         elif param.norm()>10000:
                #             print(name, param.norm())

            loss += local_trn_metrics[-2].sum()
            correct += local_trn_metrics[-1].sum()
            total += len(trn_dl.dataset)

            correct_by_class += local_trn_metrics[:self.num_classes].sum(dim=1)
            total_by_class += local_trn_metrics[self.num_classes: 2*self.num_classes].sum(dim=1)

            trn_metrics[2*ndx] = local_trn_metrics[-2].sum() / len(trn_dl.dataset)
            trn_metrics[2*ndx + 1] = local_trn_metrics[-1].sum() / len(trn_dl.dataset)

        trn_metrics[2*self.args.site_number: 2*self.args.site_number + self.num_classes] = correct_by_class / total_by_class
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

            val_metrics = torch.zeros(2 + 2*self.args.site_number + self.num_classes, device=self.device)
            loss = 0
            correct = 0
            total = 0
            correct_by_class = torch.zeros(self.num_classes, device=self.device)
            total_by_class = torch.zeros(self.num_classes, device=self.device)
            for ndx, val_dl in enumerate(val_dls):
                local_val_metrics = torch.zeros(2 + 2*self.num_classes, len(val_dl), device=self.device)

                for batch_ndx, batch_tuple in enumerate(val_dl):
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
                total += len(val_dl.dataset)

                correct_by_class += local_val_metrics[:self.num_classes].sum(dim=1)
                total_by_class += local_val_metrics[self.num_classes: 2*self.num_classes].sum(dim=1)

                val_metrics[2*ndx] = local_val_metrics[-2].sum() / len(val_dl.dataset)
                val_metrics[2*ndx + 1] = local_val_metrics[-1].sum() / len(val_dl.dataset)

            val_metrics[2*self.args.site_number: 2*self.args.site_number + self.num_classes] = correct_by_class / total_by_class
            val_metrics[-2] = loss / total
            val_metrics[-1] = correct / total

        return val_metrics.to('cpu'), correct / total, loss / total

    def computeBatchLoss(self, batch_ndx, batch_tup, model, metrics, mode, site_id):
        batch, labels = batch_tup
        # if self.args.partition == 'regular':
        #     batch = batch.to(device=self.device, non_blocking=True).float()
        # else:
        batch = batch.to(device=self.device, non_blocking=True).float().permute(0, 3, 1, 2)
        labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

        if mode == 'trn':
            assert self.args.aug_mode in ['classification', 'segmentation']
            batch = aug_image(batch)

        if not self.args.model_name == 'resnet34' and not self.args.model_name == 'resnet18':
            pred = model(batch, torch.tensor(site_id, device=self.device, dtype=torch.int))
        else:
            pred = model(batch)
        pred_label = torch.argmax(pred, dim=1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
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

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics
    ):
        self.initTensorboardWriters()

        writer = getattr(self, mode_str + '_writer')
        for ndx in range(self.args.site_number):
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
                scalar_value=metrics[2*self.args.site_number + ndx],
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
            self.args.logdir,
            '{}_{}.state'.format(
                self.time_str,
                self.args.comment
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        state = {'valmetrics':val_metrics,
                    'epoch': epoch_ndx}
        for ndx, model in enumerate(self.models):
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            if self.args.finetuning:
                state[ndx] = {
                    'model_state': model.state_dict(),
                    'trn_labels': trn_dls[ndx].dataset.labels,
                    'val_labels': val_dls[ndx].dataset.labels,
                    'trn_indices': trn_dls[ndx].dataset.indices,
                    'val_indices': val_dls[ndx].dataset.indices
                }
                if self.args.model_name == 'resnet18emb':
                    state[ndx]['emb_vector'] = model.state_dict()['embedding.weight'][ndx]
                else:
                    state[ndx]['model_state'] = model.state_dict()
            else:
                state[ndx] = {
                    'model_state': model.state_dict(),
                    'model_name': type(model).__name__,
                    'optimizer_state': self.optims[ndx].state_dict(),
                    'optimizer_name': type(self.optims[ndx]).__name__,
                    'trn_labels': trn_dls[ndx].dataset.labels,
                    'val_labels': val_dls[ndx].dataset.labels,
                    'trn_indices': trn_dls[ndx].dataset.indices,
                    'val_indices': val_dls[ndx].dataset.indices
                }

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
            if 'embedding.weight' in state_dict:
                state_dict['embedding.weight'] = state_dict['embedding.weight'][0].unsqueeze(0).repeat(self.args.site_number, 1)
            for model in self.models:
                model.load_state_dict(state_dict, strict=False)
        else:
            original_list = [name for name, _ in self.models[0].named_parameters()]
            layer_list = get_layer_list(model=self.args.model_name, strategy=self.args.strategy, original_list=original_list)
            state_dicts = [model.state_dict() for model in self.models]
            param_dict = {layer: torch.zeros(state_dicts[0][layer].shape, device=self.device) for layer in layer_list}

            for layer in layer_list:
                for state_dict in state_dicts:
                    param_dict[layer] += state_dict[layer]
                param_dict[layer] /= len(state_dicts)

            for model in self.models:
                model.load_state_dict(param_dict, strict=False)

if __name__ == '__main__':
    LayerPersonalisationTrainingApp().main()
