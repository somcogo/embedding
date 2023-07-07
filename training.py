import os
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, LBFGS
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from models.model import ResNetWithEmbeddings, ResNet34Model, ResNet18Model
from utils.logconf import logging
from utils.data_loader import get_dl_lists
from utils.ops import aug_image
from utils.merge_strategies import get_layer_list
from models.gmm import GaussianMixture

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class LayerPersonalisationTrainingApp:
    def __init__(self, epochs=2, batch_size=128, logdir='test', lr=1e-4,
                 comment='dwlpt', dataset='cifar10', site_number=1,
                 model_name='resnet18emb', optimizer_type='sgd',
                 scheduler_mode=None, pretrained=False, T_max=500,
                 label_smoothing=0.0, save_model=False, partition='regular',
                 alpha=None, strategy='all', finetuning=False, embed_dim=2,
                 model_path=None, embedding_lr=None, ffwrd_lr=None, gmm=False,
                 quick_gmm=False):

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
        self.T_max = T_max
        self.label_smoothing = label_smoothing
        self.save_model = save_model
        self.strategy = strategy
        self.finetuning = finetuning
        self.embed_dim = embed_dim
        self.model_path = model_path
        self.gmm = gmm
        self.quick_gmm = quick_gmm
        self.time_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.logdir_name)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.models = self.initModels(embed_dim=embed_dim)
        self.mergeModels(is_init=True, model_path=model_path)
        self.optims = self.initOptimizers(lr, finetuning, embedding_lr=embedding_lr, ffwrd_lr=ffwrd_lr)
        self.schedulers = self.initSchedulers()
        self.trn_dls, self.val_dls = self.initDls(batch_size=batch_size, partition=partition, alpha=alpha)
        self.trn_vector_model, self.val_vector_model, self.trn_vector_optim, self.val_vector_optim = self.initEmbeddingVector()
        self.trn_gmms, self.val_gmms = self.initGMMs()

    def initModels(self, embed_dim):
        if self.dataset == 'cifar10':
            num_classes = 10
            in_channels = 3
        elif self.dataset == 'cifar100':
            num_classes = 100
            in_channels = 3
        elif self.dataset == 'pascalvoc':
            num_classes = 21
            in_channels = 3
        elif self.dataset == 'mnist':
            num_classes = 10
            in_channels = 1
        elif self.dataset == 'imagenet':
            num_classes = 200
            in_channels = 3
        self.num_classes = num_classes
        models = []
        for _ in range(self.site_number):
            if self.model_name == 'resnet34emb':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[3, 4, 6, 3], site_number=self.site_number, embed_dim=embed_dim)
            elif self.model_name == 'resnet18emb':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim)
            elif self.model_name == 'resnet18embhypnn1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=1)
            elif self.model_name == 'resnet18embhypnn2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=2)
            elif self.model_name == 'resnet18lightweight1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True)
            elif self.model_name == 'resnet18lightweight2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True)
            elif self.model_name == 'resnet18affine1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True)
            elif self.model_name == 'resnet18affine2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True)
            elif self.model_name == 'resnet18medium1':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=1, lightweight=True, affine=True, medium_ffwrd=True)
            elif self.model_name == 'resnet18medium2':
                model = ResNetWithEmbeddings(num_classes=num_classes, in_channels=in_channels, layers=[2, 2, 2, 2], site_number=self.site_number, embed_dim=embed_dim, use_hypnns=True, version=2, lightweight=True, affine=True, medium_ffwrd=True)
            elif self.model_name == 'resnet34':
                model = ResNet34Model(num_classes=num_classes, in_channels=in_channels, pretrained=self.pretrained)
            elif self.model_name == 'resnet18':
                model = ResNet18Model(num_classes=num_classes, in_channels=in_channels, pretrained=self.pretrained)
            models.append(model)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            for model in models:
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
                layer_list = get_layer_list(self.model_name, strategy=self.strategy)
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
                    scheduler = CosineAnnealingLR(optim, T_max=self.T_max)
                schedulers.append(scheduler)
            
        return schedulers

    def initDls(self, batch_size, partition, alpha):
        index_dict = torch.load('models/{}_saved_index_maps.pt'.format(self.dataset)) if partition == 'given' else None
        trn_idx_map = index_dict[self.site_number][alpha]['trn'] if index_dict is not None else None
        val_idx_map = index_dict[self.site_number][alpha]['val'] if index_dict is not None else None
        trn_dls, val_dls = get_dl_lists(dataset=self.dataset, partition=partition, n_site=self.site_number, batch_size=batch_size, alpha=alpha, net_dataidx_map_train=trn_idx_map, net_dataidx_map_test=val_idx_map)
        return trn_dls, val_dls

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.comment)
            
    def initEmbeddingVector(self):
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

        trn_vector_model = self.initModels(self.embed_dim)[0]
        trn_vector_model.embedding = nn.Embedding(trn_size, embedding_dim=self.embed_dim, device=self.device)
        for name, param in trn_vector_model.named_parameters():
            if name != 'embedding.weight':
                param.requires_grad = False

        trn_vector_optim = Adam([trn_vector_model.embedding.weight], lr=0.001, betas=(.5,.9))
        
        val_vector_model = self.initModels(self.embed_dim)[0]
        val_vector_model.embedding = nn.Embedding(val_size, embedding_dim=self.embed_dim, device=self.device)
        for name, param in val_vector_model.named_parameters():
            if name != 'embedding.weight':
                param.requires_grad = False

        val_vector_optim = Adam([val_vector_model.embedding.weight], lr=0.001, betas=(.5,.9))

        return trn_vector_model, val_vector_model, trn_vector_optim, val_vector_optim
    
    def initGMMs(self):
        trn_gmms = []
        val_gmms = []
        for _ in range(self.site_number):
            trn_gmms.append(GaussianMixture(n_components=1, n_features=self.embed_dim).cuda())
            val_gmms.append(GaussianMixture(n_components=1, n_features=self.embed_dim).cuda())
        return trn_gmms, val_gmms

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

                if self.save_model and (accuracy==saving_criterion or self.finetuning):
                    self.saveModel(epoch_ndx, val_metrics, trn_dls, val_dls)

                if epoch_ndx < 501 or epoch_ndx % 100 == 0:
                    log.info('Epoch {} of {}, accuracy/miou {}, val loss {}'.format(epoch_ndx, self.epochs, accuracy, loss))

            if self.gmm:
                self.updateEmbeddingVectors(trn_dls, val_dls)
                self.fitGMM(epoch_ndx)
            
            
            if self.scheduler_mode == 'cosine':
                for scheduler in self.schedulers:
                    scheduler.step()
                    # log.debug(self.scheduler.get_last_lr())

            if self.site_number > 1 and not self.finetuning:
                self.mergeModels()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

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
                        if self.optimizer_type == 'lbfgs':
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

            val_metrics[2*self.site_number: 2*self.site_number + self.num_classes] = correct_by_class / total_by_class
            val_metrics[-2] = loss / total
            val_metrics[-1] = correct / total

        return val_metrics.to('cpu'), correct / total, loss / total

    def computeBatchLoss(self, batch_ndx, batch_tup, model, metrics, mode, site_id):
        batch, labels, img_id = batch_tup
        batch = batch.to(device=self.device, non_blocking=True).float().permute(0, 3, 1, 2)
        labels = labels.to(device=self.device, non_blocking=True).to(dtype=torch.long)

        if mode == 'trn':
            batch = aug_image(batch, self.dataset)

        if not self.model_name == 'resnet34' and not self.model_name == 'resnet18':
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
    
    def updateEmbeddingVectors(self, trn_dls, val_dls):
        self.copyModelToVectorModel()
        trn_vectors, val_vectors = self.extractEmbeddingVectors()

        for i in range(self.site_number):
            for batch_tup in trn_dls[i]:
                self.trn_vector_optim.zero_grad()
                imgs, labels, img_ids = batch_tup
                imgs = imgs.to(device=self.device).float().permute(0, 3, 1, 2)
                labels = labels.to(device=self.device).to(dtype=torch.long)
                img_ids = img_ids.to(device=self.device).to(dtype=torch.long)
                preds = torch.zeros(imgs.shape[0], 10, device=self.device)
                if self.quick_gmm:
                    preds = self.trn_vector_model(imgs, img_ids)
                else:
                    for j in range(imgs.shape[0]):
                        preds[j] = self.trn_vector_model(imgs[j].unsqueeze(0), img_ids[j])
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(preds, labels)
                loss.backward()
            log.debug('done with trn-site {}'.format(i))

        for i in range(self.site_number):
            for batch_tup in val_dls[i]:
                self.val_vector_optim.zero_grad()
                imgs, labels, img_ids = batch_tup
                imgs = imgs.to(device=self.device).float().permute(0, 3, 1, 2)
                labels = labels.to(device=self.device).to(dtype=torch.long)
                img_ids = img_ids.to(device=self.device).to(dtype=torch.long)
                preds = torch.zeros(imgs.shape[0], 10, device=self.device)
                if self.quick_gmm:
                    preds = self.val_vector_model(imgs, img_ids)
                else:
                    for j in range(imgs.shape[0]):
                        preds[j] = self.val_vector_model(imgs[j].unsqueeze(0), img_ids[j])
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(preds, labels)
                loss.backward()
            log.debug('done with val-site {}'.format(i))
        self.trn_vector_optim.step()
        self.val_vector_optim.step()
        log.debug('completed vector update')
    
    def fitGMM(self, epoch_ndx):
        trn_vectors, val_vectors = self.extractEmbeddingVectors()
        trn_img_indices, val_img_indices = self.getImageIndicesBySite()

        state = {'trn':{}, 'val':{}}
        for i, indices in enumerate(trn_img_indices):
            self.trn_gmms[i].fit(trn_vectors[indices])
            state['trn'][i] = {'vectors':trn_vectors[indices],
                               'pred':self.trn_gmms[i].predict(trn_vectors),
                               'mu':self.trn_gmms[i].mu,
                               'var':self.trn_gmms[i].var}
        for i, indices in enumerate(val_img_indices):
            self.val_gmms[i].fit(val_vectors[indices])
            state['val'][i] = {'vectors':val_vectors[indices],
                               'pred':self.val_gmms[i].predict(val_vectors),
                               'mu':self.val_gmms[i].mu,
                               'var':self.val_gmms[i].var}
        save_path = os.path.join(
            'gmm_data',
            self.logdir_name,
            '{}_{}_{}.pt'.format(self.time_str, self.comment, epoch_ndx))
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
            '{}_{}.state'.format(
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
            if self.finetuning:
                state[ndx] = {
                    'model_state': model.state_dict(),
                    'trn_labels': trn_dls[ndx].dataset.labels,
                    'val_labels': val_dls[ndx].dataset.labels,
                    'trn_indices': trn_dls[ndx].dataset.indices,
                    'val_indices': val_dls[ndx].dataset.indices
                }
                if self.model_name == 'resnet18emb':
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
