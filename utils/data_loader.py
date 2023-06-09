from torch.utils.data import DataLoader

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_mnist_datasets, get_image_net_dataset, TruncatedDataset
from .partition import partition_by_class, partition_with_dirichlet_distribution

data_path = 'data/'

def get_datasets(data_dir, dataset):
    if dataset == 'cifar10':
        trn_dataset, val_dataset = get_cifar10_datasets(data_dir=data_dir)
    elif dataset == 'cifar100':
        trn_dataset, val_dataset = get_cifar100_datasets(data_dir=data_dir)
    elif dataset == 'mnist':
        trn_dataset, val_dataset = get_mnist_datasets(data_dir=data_dir)
    elif dataset == 'imagenet':
        trn_dataset, val_dataset = get_image_net_dataset(data_dir=data_dir)
    return trn_dataset, val_dataset

def get_dl_lists(dataset, batch_size, partition=None, n_site=None, alpha=None, net_dataidx_map_train=None, net_dataidx_map_test=None):
    trn_dataset, val_dataset = get_datasets(data_dir=data_path, dataset=dataset)

    if partition == 'regular':
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset) for _ in range(n_site)]
        val_ds_list = [TruncatedDataset(val_dataset, dataset) for _ in range(n_site)]
    elif partition == 'by_class':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_by_class(data_dir=data_path, dataset=dataset, n_sites=n_site)
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'dirichlet':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_with_dirichlet_distribution(data_dir=data_path, dataset=dataset, n_sites=n_site, alpha=alpha)
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]
    elif partition == 'given':
        trn_ds_list = [TruncatedDataset(trn_dataset, dataset, idx_map) for idx_map in net_dataidx_map_train.values()]
        val_ds_list = [TruncatedDataset(val_dataset, dataset, idx_map) for idx_map in net_dataidx_map_test.values()]

    trn_dl_list = [DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=True, drop_last=True) for trn_ds in trn_ds_list]
    val_dl_list = [DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, drop_last=True) for val_ds in val_ds_list]
    return trn_dl_list, val_dl_list