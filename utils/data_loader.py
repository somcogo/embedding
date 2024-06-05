from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_mnist_datasets, get_image_net_dataset, get_celeba_dataset, get_minicoco_dataset
from .partition import partition_by_class, partition_with_dirichlet_distribution, cont_partition

data_path = '/home/hansel/developer/embedding/data/'

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
    return trn_dataset, val_dataset

def get_dl_lists(dataset, batch_size, partition=None, n_site=None, alpha=None, net_dataidx_map_train=None, net_dataidx_map_test=None, shuffle=True, seed=None, site_indices=None, use_hdf5=True, cross_val_id=None):
    trn_dataset, val_dataset = get_datasets(data_dir=data_path, dataset=dataset, use_hdf5=use_hdf5)

    if partition == 'regular':
        trn_ds_list = [Subset(trn_dataset, range(len(trn_dataset))) for _ in range(n_site)]
        val_ds_list = [Subset(val_dataset, range(len(val_dataset))) for _ in range(n_site)]
    elif partition == 'by_class':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_by_class(data_dir=data_path, dataset=dataset, n_sites=n_site, seed=seed)
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'dirichlet':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_with_dirichlet_distribution(data_dir=data_path, dataset=dataset, n_sites=n_site, alpha=alpha, seed=seed)
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'cont':
        (net_dataidx_map_train, net_dataidx_map_test) = cont_partition(data_dir=data_path, dataset=dataset, n_sites=n_site)
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'given':
        trn_ds_list = [Subset(trn_dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [Subset(val_dataset, idx_map) for idx_map in net_dataidx_map_test.values()]

    if cross_val_id is not None:
        merged_ds_list = [ConcatDataset([trn_ds, val_ds]) for trn_ds, val_ds in zip(trn_ds_list, val_ds_list)]
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        splits = [list(kfold.split(range(len(merged_ds)))) for merged_ds in merged_ds_list]
        indices = [split[cross_val_id] for split in splits]
        trn_ds_list = [Subset(ds, idx_map[0]) for ds, idx_map in zip(merged_ds_list, indices)]
        val_ds_list = [Subset(ds, idx_map[1]) for ds, idx_map in zip(merged_ds_list, indices)]

    trn_dl_list = [DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0) for val_ds in val_ds_list]
    if site_indices is not None:
        trn_dl_list = [trn_dl_list[i] for i in site_indices]
        val_dl_list = [val_dl_list[i] for i in site_indices]
    return trn_dl_list, val_dl_list

def new_get_dl_lists(dataset, batch_size, degradation, n_site=None, seed=None):
    if degradation == 'mixed':
        trn_l1, val_l1 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 2, alpha=1e7, seed=seed)
        trn_l2, val_l2 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='cont', n_site=n_site // 2, seed=seed)
        trn_dl_list = trn_l1 + trn_l2
        val_dl_list = val_l1 + val_l2
    elif degradation == '3mixed':
        trn_l1, val_l1 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 3, alpha=1e7, seed=seed)
        trn_l2, val_l2 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='cont', n_site=n_site // 3, seed=seed)
        trn_l3, val_l3 = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site // 3, alpha=1e7, seed=seed)
        trn_dl_list = trn_l1 + trn_l2 + trn_l3
        val_dl_list = val_l1 + val_l2 + val_l3
    elif degradation in ['addgauss', 'jittermix', 'jitter', 'randgauss']:
        trn_dl_list, val_dl_list = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='dirichlet', n_site=n_site, alpha=1e7, seed=seed)
    elif degradation == 'classskew':
        trn_dl_list, val_dl_list = get_dl_lists(dataset=dataset, batch_size=batch_size, partition='cont', n_site=n_site, seed=seed)

    return trn_dl_list, val_dl_list