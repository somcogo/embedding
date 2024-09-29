import random

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_mnist_datasets, get_image_net_dataset, get_celeba_dataset, get_minicoco_dataset, get_digits_dataset
from .partition import partition_wrap, new_partition_wrap

data_path = '/home/hansel/developer/embedding/data/'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_datasets(data_dir, dataset, use_hdf5=False):
    if dataset == 'cifar10':
        trn_dataset, val_dataset = get_cifar10_datasets(data_dir=data_dir, use_hdf5=use_hdf5)
    elif dataset == 'cifar100':
        trn_dataset, val_dataset = get_cifar100_datasets(data_dir=data_dir)
    elif dataset == 'mnist':
        trn_dataset, val_dataset = get_mnist_datasets(data_dir=data_dir, use_hdf5=use_hdf5)
    elif dataset == 'imagenet':
        trn_dataset, val_dataset = get_image_net_dataset(data_dir=data_dir)
    elif dataset == 'celeba':
        trn_dataset, val_dataset = get_celeba_dataset(data_dir=data_dir)
    elif dataset == 'minicoco':
        trn_dataset, val_dataset = get_minicoco_dataset(data_dir=data_dir)
    elif dataset == 'digits':
        trn_dataset, val_dataset = get_digits_dataset(data_dir=data_dir)
    return trn_dataset, val_dataset

def get_dl_lists(dataset, batch_size, partition=None, n_site=None, alpha=None, net_dataidx_map_train=None, net_dataidx_map_test=None, shuffle=True, seed=None, site_indices=None, use_hdf5=True, cross_val_id=None, gl_seed=None, cl_per_site=None):
    trn_dataset, val_dataset = get_datasets(data_dir=data_path, dataset=dataset, use_hdf5=use_hdf5)

    g = torch.Generator()
    loader_seed = gl_seed if gl_seed is not None else 0
    g.manual_seed(loader_seed)

    if partition == 'regular':
        trn_ds_list = [Subset(trn_dataset, range(len(trn_dataset))) for _ in range(n_site)]
        val_ds_list = [Subset(val_dataset, range(len(val_dataset))) for _ in range(n_site)]
    elif partition == 'given':
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    else:
        (net_dataidx_map_train, net_dataidx_map_test) = partition_wrap(data_dir=data_path, dataset=dataset, partition=partition, n_sites=n_site, alpha=alpha, seed=seed, cl_per_site=cl_per_site)
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in net_dataidx_map_test.values()]

    if cross_val_id is not None:
        merged_ds_list = [ConcatDataset([trn_ds, val_ds]) for trn_ds, val_ds in zip(trn_ds_list, val_ds_list)]
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        splits = [list(kfold.split(range(len(merged_ds)))) for merged_ds in merged_ds_list]
        indices = [split[cross_val_id] for split in splits]
        trn_ds_list = [Subset(ds, idx_map[0]) for ds, idx_map in zip(merged_ds_list, indices)]
        val_ds_list = [Subset(ds, idx_map[1]) for ds, idx_map in zip(merged_ds_list, indices)]

    trn_dl_list = [DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0, worker_init_fn=seed_worker, generator=g) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=seed_worker, generator=g) for val_ds in val_ds_list]
    if site_indices is not None:
        trn_dl_list = [trn_dl_list[i] for i in site_indices]
        val_dl_list = [val_dl_list[i] for i in site_indices]
    return trn_dl_list, val_dl_list

def deg_list_dl_list(dataset, batch_size, degradations, n_site=None, seed=None):
    site_per_deg = n_site // len(degradations)
    trn_dls, val_dls = [], []
    for deg in degradations:
        t, v = new_get_dl_lists(dataset=dataset, batch_size=batch_size, degradation=deg, n_site=site_per_deg, seed=seed)
        trn_dls.append(t)
        val_dls.append(v)
    return sum(trn_dls, []), sum(val_dls, [])

def refactored_get_dls(dataset, batch_size, degs, n_sites, alpha=None, trn_map=None, val_map=None, shuffle=True, seed=None, site_indices=None, use_hdf5=True, cross_val_id=None, gl_seed=None, cl_per_site=None, trn_set_size=None):
    trn_dataset, val_dataset = get_datasets(data_dir=data_path, dataset=dataset, use_hdf5=use_hdf5)

    g = torch.Generator()
    loader_seed = gl_seed if gl_seed is not None else 0
    g.manual_seed(loader_seed)

    if degs == 'regular':
        trn_ds_list = [Subset(trn_dataset, range(len(trn_dataset))) for _ in range(n_sites)]
        val_ds_list = [Subset(val_dataset, range(len(val_dataset))) for _ in range(n_sites)]
    elif degs == 'given':
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in trn_map.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in val_map.values()]
    else:
        (trn_map, val_map) = new_partition_wrap(data_path, dataset, degs, n_sites, alpha, seed, cl_per_site)
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in trn_map.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in val_map.values()]

    if cross_val_id is not None:
        merged_ds_list = [ConcatDataset([trn_ds, val_ds]) for trn_ds, val_ds in zip(trn_ds_list, val_ds_list)]
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        splits = [list(kfold.split(range(len(merged_ds)))) for merged_ds in merged_ds_list]
        indices = [split[cross_val_id] for split in splits]
        trn_ds_list = [Subset(ds, idx_map[0]) for ds, idx_map in zip(merged_ds_list, indices)]
        val_ds_list = [Subset(ds, idx_map[1]) for ds, idx_map in zip(merged_ds_list, indices)]

    if trn_set_size is not None:
        trn_ds_list = [Subset(ds, np.arange(trn_set_size)) for ds in trn_ds_list]

    trn_dl_list = [DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0, worker_init_fn=seed_worker, generator=g) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(dataset=val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=seed_worker, generator=g) for val_ds in val_ds_list]
    if site_indices is not None:
        trn_dl_list = [trn_dl_list[i] for i in site_indices]
        val_dl_list = [val_dl_list[i] for i in site_indices]
    return trn_dl_list, val_dl_list

def new_get_dl_lists(dataset, batch_size, degradation, n_site=None, seed=None, cross_val_id=None, gl_seed=None, cl_per_site=None):
    if degradation == 'mixed':
        trn_l1, val_l1 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 2, alpha=1e7, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_l2, val_l2 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='cont', n_site=n_site // 2, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_dl_list = trn_l1 + trn_l2
        val_dl_list = val_l1 + val_l2
    elif degradation == 'mixedv2':
        trn_l1, val_l1 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 2, alpha=1e7, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_l2, val_l2 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='classshard', n_site=n_site // 2, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed, cl_per_site=cl_per_site)
        trn_dl_list = trn_l1 + trn_l2
        val_dl_list = val_l1 + val_l2
    elif degradation == '3mixed':
        trn_l1, val_l1 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 3, alpha=1e7, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_l2, val_l2 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='cont', n_site=n_site // 3, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_l3, val_l3 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 3, alpha=1e7, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_dl_list = trn_l1 + trn_l2 + trn_l3
        val_dl_list = val_l1 + val_l2 + val_l3
    elif degradation == '3mixedv2':
        trn_l1, val_l1 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 3, alpha=1e7, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_l2, val_l2 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='classshard', n_site=n_site // 3, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed, cl_per_site=cl_per_site)
        trn_l3, val_l3 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 3, alpha=1e7, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
        trn_dl_list = trn_l1 + trn_l2 + trn_l3
        val_dl_list = val_l1 + val_l2 + val_l3
    elif degradation in ['addgauss', 'jittermix', 'jitter', 'randgauss', 'colorjitter']:
        trn_dl_list, val_dl_list = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site, alpha=1e7, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
    elif degradation == 'classskew':
        trn_dl_list, val_dl_list = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='cont', n_site=n_site, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
    elif degradation == 'classsep':
        trn_dl_list, val_dl_list = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='by_class', n_site=n_site, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
    elif degradation == 'classshard':
        trn_dl_list, val_dl_list = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='classshard', n_site=n_site, seed=seed, cross_val_id=cross_val_id, gl_seed=gl_seed, cl_per_site=cl_per_site)

    return trn_dl_list, val_dl_list