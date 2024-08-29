from collections import defaultdict
import random

import numpy as np

from .datasets import get_cifar10_datasets, get_cifar100_datasets, get_mnist_datasets, get_image_net_dataset, get_celeba_dataset, get_minicoco_dataset, get_digits_dataset

def partition_wrap(data_dir, dataset, partition, n_sites, alpha=None, seed=None, cl_per_site=None):

    if partition == 'by-class':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_by_class(data_dir, dataset, n_sites, seed)
    elif partition == 'dirichlet':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_with_dirichlet_distribution(data_dir, dataset, n_sites, alpha, seed)
    elif partition == 'cont':
        (net_dataidx_map_train, net_dataidx_map_test) = cont_partition(data_dir, dataset, n_sites)
    elif partition == 'classshard':
        (net_dataidx_map_train, net_dataidx_map_test) = partition_with_shards(data_dir, dataset, n_sites, seed, cl_per_site)
    return (net_dataidx_map_train, net_dataidx_map_test)

def get_classes(data_dir, dataset):
    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir)
        K = 10
    elif dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
        K = 100
    elif dataset == 'mnist':
        train_ds, test_ds = get_mnist_datasets(data_dir)
        K = 10
    elif dataset == 'imagenet':
        train_ds, test_ds = get_image_net_dataset(data_dir)
        K = 200

    if dataset in ['imagenet']:
        trn_classes = train_ds.labels
        val_classes = test_ds.labels
    else:
        trn_classes = train_ds.targets
        val_classes = test_ds.targets
    return K, trn_classes, val_classes


def new_partition_wrap(data_dir, dataset, degs, n_sites, alpha=None, seed=None, cl_per_site=None):
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
            deg_trn_map, deg_val_map = new_partition(trn_idx_by_deg[i], val_idx_by_deg[i], trn_classes[trn_idx_by_deg[i]], val_classes[val_idx_by_deg[i]], deg, site_per_deg, alpha, seed, cl_per_site, cl_num)
            for j in range(site_per_deg):
                trn_map[site_per_deg*i + j] = deg_trn_map[j]
                val_map[site_per_deg*i + j] = deg_val_map[j]

    return trn_map, val_map

def new_partition(trn_indices, val_indices, trn_classes, val_classes, deg, n_sites, alpha, seed, cl_per_site, cl_num):
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
    elif deg == 'dirichlet':
        trn_map, val_map = new_partition_dirichlet(trn_indices, val_indices, trn_classes, val_classes, n_sites, cl_num, alpha, seed)
    else:
        trn_map, val_map = new_partition_dirichlet(trn_indices, val_indices, trn_classes, val_classes, n_sites, cl_num, 1e7, seed)
    return trn_map, val_map

def partition_digits(data_dir, n_sites, seed):
    trn_ds, val_ds = get_digits_dataset(data_dir)
    # trn_lengths = [60000, 7291, 73257, 10000]
    # val_lengths = [10000, 2007, 26032, 2000]
    # [ 60000,  67291, 140548, 150548], [10000, 12007, 38039, 40039]
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

def partition_with_shards(data_dir, dataset, n_sites, seed=None, cl_per_site=None):
    cl_per_site = 2 if cl_per_site is None else cl_per_site
    rng = np.random.default_rng(seed)
    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir)
        K = 10
    elif dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
        K = 100
    elif dataset == 'mnist':
        train_ds, test_ds = get_mnist_datasets(data_dir)
        K = 10
    elif dataset == 'imagenet':
        train_ds, test_ds = get_image_net_dataset(data_dir)
        K = 200
    shards = cl_per_site * n_sites // K

    if dataset in ['imagenet']:
        y_train = train_ds.labels
        y_test = test_ds.labels
    else:
        y_train = train_ds.targets
        y_test = test_ds.targets

    train_shards = {i: np.split(rng.permutation(np.where(y_train == i)[0]), shards) for i in range(K)}
    test_shards = {i: np.split(rng.permutation(np.where(y_test == i)[0]), shards) for i in range(K)}

    class_count = np.full((K), shards)

    trn_map = {}
    val_map = {}
    for i in range(n_sites):
        trn_map[i] = []
        val_map[i] = []
        for _ in range(cl_per_site):
            max_class_counts = np.where(np.array(class_count) == max(class_count))[0]
            choice = rng.choice(max_class_counts)
            trn_map[i].append(train_shards[choice][max(class_count) - 1])
            val_map[i].append(test_shards[choice][max(class_count) - 1])
            class_count[choice] -= 1
    for i in range(n_sites):
        trn_map[i] = np.concatenate(trn_map[i], axis=0)
        val_map[i] = np.concatenate(val_map[i], axis=0)

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

def cont_partition(data_dir, dataset, n_sites):
    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir)
        K = 10
    elif dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
        K = 100
    elif dataset == 'mnist':
        train_ds, test_ds = get_mnist_datasets(data_dir)
        K = 10
    elif dataset == 'imagenet':
        train_ds, test_ds = get_image_net_dataset(data_dir)
        K = 200
    elif dataset == 'celeba':
        train_ds, test_ds = get_celeba_dataset(data_dir)
        K = 18
    elif dataset == 'minicoco':
        train_ds, test_ds = get_minicoco_dataset(data_dir)
        K = 12
    assert n_sites <= K

    if dataset in ['imagenet']:
        y_train = train_ds.labels
        y_test = test_ds.labels
    else:
        y_train = train_ds.targets
        y_test = test_ds.targets
    
    
    trn_map = {i: np.where(y_train <= i)[0] for i in range(K)}
    val_map = {i: np.where(y_test <= i)[0] for i in range(K)}

    return trn_map, val_map
    

def partition_by_class(data_dir, dataset, n_sites, seed=None):
    rng = np.random.default_rng(seed)

    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir)
        K = 10
        num = K // n_sites
    elif dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
        K = 100
        num = K // n_sites
    elif dataset == 'mnist':
        train_ds, test_ds = get_mnist_datasets(data_dir)
        K = 10
        num = K // n_sites
    elif dataset == 'imagenet':
        train_ds, test_ds = get_image_net_dataset(data_dir)
        K = 200
        num = K // n_sites
    elif dataset == 'celeba':
        train_ds, test_ds = get_celeba_dataset(data_dir)
        K = 18
        num = K // n_sites
    elif dataset == 'minicoco':
        train_ds, test_ds = get_minicoco_dataset(data_dir)
        K = 12
        num = K // n_sites

    if dataset in ['celeba', 'imagenet', 'minicoco']:
        y_train = train_ds.labels
        y_test = test_ds.labels
    else:
        y_train = train_ds.targets
        y_test = test_ds.targets

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    assert (num * n_sites) % K == 0, "equal classes appearance is needed"
    count_per_class = (num * n_sites) // K
    class_dict = {}
    for i in range(K):
        # sampling alpha_i_c
        probs = rng.uniform(0.4, 0.6, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(n_sites):
        c = []
        for _ in range(num):
            class_counts = [class_dict[i]['count'] for i in range(K)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(rng.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    if dataset == 'celeba':
        data_class_idx_train = {i: np.where([pres_cl[i] for pres_cl in y_train])[0] for i in range(K)}
        data_class_idx_test = {i: np.where([pres_cl[i] for pres_cl in y_test])[0] for i in range(K)}
    else:
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

    num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
    num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx_train.values():
        rng.shuffle(data_idx)
    for data_idx in data_class_idx_test.values():
        rng.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_sites)}
    net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_sites)}

    for usr_i in range(n_sites):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx_train = int(num_samples_train[c] * p)
            end_idx_test = int(num_samples_test[c] * p)

            net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i], data_class_idx_train[c][:end_idx_train])
            net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i], data_class_idx_test[c][:end_idx_test])

            data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
            data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    return (net_dataidx_map_train, net_dataidx_map_test)

# TODO: link fedml, fedtp 
# https://github.com/zhyczy/FedTP/blob/main/utils.py
# https://github.com/FedML-AI/FedML/blob/master/python/fedml/core/data/noniid_partition.py

def partition_with_dirichlet_distribution(data_dir, dataset, n_sites, alpha, seed=None):
    rng = np.random.default_rng(seed)

    if dataset == 'cifar10':
        train_ds, test_ds = get_cifar10_datasets(data_dir, use_hdf5=True)
        K = 10
    elif dataset == 'cifar100':
        train_ds, test_ds = get_cifar100_datasets(data_dir)
        K = 100
    elif dataset == 'mnist':
        train_ds, test_ds = get_mnist_datasets(data_dir, use_hdf5=True)
        K = 10
    elif dataset == 'imagenet':
        train_ds, test_ds = get_image_net_dataset(data_dir)
        K = 200
    elif dataset == 'celeba':
        train_ds, test_ds = get_celeba_dataset(data_dir)
        K = 18
    elif dataset == 'minicoco':
        train_ds, test_ds = get_minicoco_dataset(data_dir)
        K = 12

    if dataset in ['celeba', 'imagenet', 'minicoco']:
        y_train = train_ds.labels
        y_test = test_ds.labels
    else:
        y_train = train_ds.targets
        y_test = test_ds.targets

    min_size = 0
    min_require_size = 10

    N_train = len(y_train)
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_sites)]
        idx_batch_test = [[] for _ in range(n_sites)]
        for k in range(K):
            if dataset in ['celeba', 'minicoco']:
                if k == 0:
                    train_idx_k = y_train[:, 0]
                    test_idx_k = y_test[:, 0]
                    # train_idx_k = np.asarray([np.any(y_train[i] == k) for i in range(len(y_train))])
                    # test_idx_k = np.asarray([np.any(y_test[i] == k) for i in range(len(y_test))])
                else:
                    train_idx_k = np.logical_and(y_train[:, k], np.logical_not(y_train[:, :k].any(axis=1)))
                    test_idx_k = np.logical_and(y_test[:, k], np.logical_not(y_test[:, :k].any(axis=1)))
                    # train_idx_k = np.asarray([np.any(y_train[i] == k)
                    #                             and not np.any(np.in1d(y_train[i], range(k - 1)))
                    #                             for i in range(len(y_train))])
                    # test_idx_k = np.asarray([np.any(y_test[i] == k)
                    #                             and not np.any(np.in1d(y_test[i], range(k - 1)))
                    #                             for i in range(len(y_test))])
                train_idx_k = np.where(train_idx_k)[0]
                test_idx_k = np.where(test_idx_k)[0]
            else:
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]
            
            idx_batch_train, idx_batch_test, min_size = partition_class_samples_with_dirichlet_distribution(N_train, alpha, n_sites, idx_batch_train, idx_batch_test, train_idx_k, test_idx_k, rng)
        
    for j in range(n_sites):
        rng.shuffle(idx_batch_train[j])
        rng.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    return (net_dataidx_map_train, net_dataidx_map_test)

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
