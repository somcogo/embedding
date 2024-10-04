from collections import defaultdict
import random

import numpy as np

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_digits_dataset

def get_classes(data_dir, dataset):
    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir)
        K = 10
    elif dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
        K = 100

    if dataset in ['imagenet']:
        trn_classes = train_ds.labels
        val_classes = test_ds.labels
    else:
        trn_classes = train_ds.targets
        val_classes = test_ds.targets
    return K, trn_classes, val_classes


def new_partition_wrap(data_dir, dataset, degs, n_sites, seed=None, cl_per_site=None):
    if type(degs) != list:
        degs = [degs]
    if dataset == 'digits':
        trn_map, val_map = partition_digits(data_dir, n_sites, seed)
    else:
        cl_num, trn_classes, val_classes = get_classes(data_dir, dataset)
        trn_idx_by_deg, val_idx_by_deg = new_partition_with_shards(np.arange(len(trn_classes)), np.arange(len(val_classes)), trn_classes, val_classes, n_sites=len(degs), cl_num=cl_num, seed=seed, cl_per_site=cl_num)

        assert n_sites % len(degs) == 0
        site_per_deg = n_sites // len(degs)
        trn_map, val_map = {}, {}
        for i, deg in enumerate(degs):
            deg_trn_map, deg_val_map = new_partition(trn_idx_by_deg[i], val_idx_by_deg[i], trn_classes[trn_idx_by_deg[i]], val_classes[val_idx_by_deg[i]], deg, site_per_deg, seed, cl_per_site, cl_num)
            for j in range(site_per_deg):
                trn_map[site_per_deg*i + j] = deg_trn_map[j]
                val_map[site_per_deg*i + j] = deg_val_map[j]

    return trn_map, val_map

def new_partition(trn_indices, val_indices, trn_classes, val_classes, deg, n_sites, seed, cl_per_site, cl_num):
    if deg == 'classshard':
        trn_map, val_map = new_partition_with_shards(trn_indices, val_indices, trn_classes, val_classes, n_sites, cl_num, seed, cl_per_site)
    elif deg == 'alphascale':
        trn_map_per_alpha, val_map_per_alpha = new_partition_with_shards(trn_indices, val_indices, trn_classes, val_classes, n_sites=n_sites//2, cl_num=cl_num, cl_per_site=cl_num, seed=seed)
        alphas = np.logspace(-1, 1, n_sites//2)
        trn_map, val_map = {}, {}
        for i in range(n_sites//2):
            alpha_trn_classes = np.array([trn_classes[np.where(trn_indices == ndx)[0][0]] for ndx in trn_map_per_alpha[i]])
            alpha_val_classes = np.array([val_classes[np.where(val_indices == ndx)[0][0]] for ndx in val_map_per_alpha[i]])
            alpha_trn_map, alpha_val_map = new_partition_dirichlet(trn_map_per_alpha[i], val_map_per_alpha[i], alpha_trn_classes, alpha_val_classes, 2, cl_num, alphas[i], seed)
            trn_map[2*i] = alpha_trn_map[0]
            trn_map[2*i+1] = alpha_trn_map[1]
            val_map[2*i] = alpha_val_map[0]
            val_map[2*i+1] = alpha_val_map[1]
    else:
        trn_map, val_map = new_partition_dirichlet(trn_indices, val_indices, trn_classes, val_classes, n_sites, cl_num, 1e7, seed)
    return trn_map, val_map

def partition_digits(data_dir, n_sites, seed):
    trn_ds, val_ds = get_digits_dataset(data_dir)
    m_trn_ndx, m_val_ndx = np.arange(60000), np.arange(10000)
    u_trn_ndx, u_val_ndx = np.arange(60000, 67291), np.arange(10000, 12007)
    sv_trn_ndx, sv_val_ndx = np.arange(67291, 140548), np.arange(12007, 38039)
    sy_trn_ndx, sy_val_ndx = np.arange(140548, 150548), np.arange(38039, 40039)
    sites_per_domain = n_sites // 4

    per_digit_maps = []
    per_digit_maps.append(new_partition_dirichlet(m_trn_ndx, m_val_ndx, trn_ds.datasets[0].targets, val_ds.datasets[0].targets, sites_per_domain, cl_num=10, alpha=1e7, seed=seed))
    per_digit_maps.append(new_partition_dirichlet(u_trn_ndx, u_val_ndx, trn_ds.datasets[1].targets, val_ds.datasets[1].targets, sites_per_domain, cl_num=10, alpha=1e7, seed=seed))
    per_digit_maps.append(new_partition_dirichlet(sv_trn_ndx, sv_val_ndx, trn_ds.datasets[2].targets, val_ds.datasets[2].targets, sites_per_domain, cl_num=10, alpha=1e7, seed=seed))
    per_digit_maps.append(new_partition_dirichlet(sy_trn_ndx, sy_val_ndx, trn_ds.datasets[3].targets, val_ds.datasets[3].targets, sites_per_domain, cl_num=10, alpha=1e7, seed=seed))

    trn_map, val_map = {}, {}
    for i, (trn_m, val_m) in enumerate(per_digit_maps):
        for j in range(sites_per_domain):
            trn_map[i * sites_per_domain + j] = trn_m[j]
            val_map[i * sites_per_domain + j] = val_m[j]

    return trn_map, val_map

def new_partition_with_shards(trn_indices, val_indices, trn_classes, val_classes, n_sites, cl_num, seed=None, cl_per_site=None):
    cl_per_site = 2 if cl_per_site is None else cl_per_site
    shards = cl_per_site * n_sites // cl_num
    rng = np.random.default_rng(seed)
    trn_shards = {i: np.array_split(trn_indices[rng.permutation(np.where(trn_classes == i)[0])], shards) for i in range(cl_num)}
    val_shards = {i: np.array_split(val_indices[rng.permutation(np.where(val_classes == i)[0])], shards) for i in range(cl_num)}

    class_count = np.full((cl_num), shards)

    trn_map = {}
    val_map = {}
    for i in range(n_sites):
        trn_map[i] = []
        val_map[i] = []
        for _ in range(cl_per_site):
            max_class_counts = np.where(np.array(class_count) == max(class_count))[0]
            choice = rng.choice(max_class_counts)
            trn_map[i].append(trn_shards[choice][max(class_count) - 1])
            val_map[i].append(val_shards[choice][max(class_count) - 1])
            class_count[choice] -= 1
    for i in range(n_sites):
        trn_map[i] = np.concatenate(trn_map[i], axis=0)
        val_map[i] = np.concatenate(val_map[i], axis=0)

    return trn_map, val_map

def new_partition_dirichlet(trn_indices, val_indices, trn_classes, val_classes, n_sites, cl_num, alpha, seed=None):
    rng = np.random.default_rng(seed)

    min_size = 0
    min_require_size = 10

    N_train = len(trn_classes)
    trn_map = {}
    val_map = {}
    while min_size < min_require_size:
        trn_idx_batch = [[] for _ in range(n_sites)]
        val_idx_batch = [[] for _ in range(n_sites)]
        for k in range(cl_num):
            trn_idx_k = trn_indices[np.where(trn_classes == k)[0]]
            val_idx_k = val_indices[np.where(val_classes == k)[0]]
            
            trn_idx_batch, val_idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N_train, alpha, n_sites, trn_idx_batch, val_idx_batch, trn_idx_k, val_idx_k, rng)
        
    for j in range(n_sites):
        rng.shuffle(trn_idx_batch[j])
        rng.shuffle(val_idx_batch[j])
        trn_map[j] = np.array(trn_idx_batch[j])
        val_map[j] = np.array(val_idx_batch[j])

    return trn_map, val_map

def partition_class_samples_with_dirichlet_distribution(N, alpha, n_sites, idx_batch_train, idx_batch_test, train_idx_k, test_idx_k, rng):

    rng.shuffle(train_idx_k)
    rng.shuffle(test_idx_k)
    
    proportions = rng.dirichlet(np.repeat(alpha, n_sites))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / n_sites) for p, idx_j in zip(proportions, idx_batch_train)]
    )
    proportions = proportions / proportions.sum()
    proportions_train = (np.cumsum(proportions) * len(train_idx_k)).round().astype(int)[:-1]
    proportions_test = (np.cumsum(proportions) * len(test_idx_k)).round().astype(int)[:-1]

    idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
    idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]

    min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
    min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
    min_size = min(min_size_train, min_size_test)

    return idx_batch_train, idx_batch_test, min_size
