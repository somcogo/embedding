from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_mnist_datasets, get_image_net_dataset, TruncatedDataset, MergedDataset
from .partition import partition_by_class, partition_with_dirichlet_distribution

data_path = 'data/'

def get_datasets(data_dir, dataset, use_hdf5=False):
    if dataset == 'cifar10':
        trn_dataset, val_dataset = get_cifar10_datasets(data_dir=data_dir, use_hdf5=use_hdf5)
    elif dataset == 'cifar100':
        trn_dataset, val_dataset = get_cifar100_datasets(data_dir=data_dir)
    elif dataset == 'mnist':
        trn_dataset, val_dataset = get_mnist_datasets(data_dir=data_dir, use_hdf5=use_hdf5)
    elif dataset == 'imagenet':
        trn_dataset, val_dataset = get_image_net_dataset(data_dir=data_dir)
    return trn_dataset, val_dataset

def get_dl_lists(dataset, batch_size, partition=None, n_site=None, alpha=None, net_dataidx_map_train=None, net_dataidx_map_test=None, shuffle=True, k_fold_val_id=None, seed=None, site_indices=None, use_hdf5=False):
    trn_dataset, val_dataset = get_datasets(data_dir=data_path, dataset=dataset, use_hdf5=use_hdf5)

    if partition == 'regular':
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset) for _ in range(n_site)]
        val_ds_list = [TruncatedDataset(val_dataset, dataset) for _ in range(n_site)]
    elif partition == 'by_class':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_by_class(data_dir=data_path, dataset=dataset, n_sites=n_site, seed=seed)
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'dirichlet':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_with_dirichlet_distribution(data_dir=data_path, dataset=dataset, n_sites=n_site, alpha=alpha, seed=seed)
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'given':
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == '5foldval':
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
        merged_ds_list = [MergedDataset(trn_ds_list[i], val_ds_list[i], dataset) for i in range(len(trn_ds_list))]
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        splits = [list(kfold.split(range(len(merged_ds)))) for merged_ds in merged_ds_list]
        indices = [split[k_fold_val_id] for split in splits]
        trn_ds_list = [TruncatedDataset(merged_ds_list[i], dataset, idx_map[0]) for i, idx_map in enumerate(indices)]
        val_ds_list = [TruncatedDataset(merged_ds_list[i], dataset, idx_map[1]) for i, idx_map in enumerate(indices)]

    trn_dl_list = [DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=shuffle, drop_last=True) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, drop_last=True) for val_ds in val_ds_list]
    if site_indices is not None:
        trn_dl_list = [trn_dl_list[i] for i in site_indices]
        val_dl_list = [val_dl_list[i] for i in site_indices]
    return trn_dl_list, val_dl_list