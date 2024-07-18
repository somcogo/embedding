import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from utils.data_loader import new_get_dl_lists
from utils.ops import get_test_transforms, get_class_list, get_ft_indices, create_mask_from_onehot, transform_image
from utils.get_model import get_model

def xygrid():
    embs = []
    for x in np.linspace(-20, 20, 25):
        for y in np.linspace(-10, 20, 25):
            embs.append(torch.tensor([x, y, 0, 0]))
    return embs

def load_embs(model_path):
    saved_dict = torch.load(model_path)
    saved_embs = torch.zeros((len(saved_dict.keys()) - 2, saved_dict[0]['site_model_state']['embedding'].shape[0]))
    for i in range(len(saved_dict.keys()) - 2):
        saved_embs[i] = saved_dict[i]['site_model_state']['embedding']
    return saved_embs

def pca_grid(model_path=None, vectors=None):
    if vectors is not None:
        saved_embs_n = vectors
    else:
        saved_embs_n = load_embs(model_path)
    pca = PCA(n_components=saved_embs_n.shape[1])
    pca.fit(saved_embs_n)
    saved_embs_pca = pca.transform(saved_embs_n)
    largest = np.linalg.norm(saved_embs_pca, axis=1).max()
    res = 10
    size = 1.5
    embs_n = np.zeros((res*res, 4))
    for i, x in enumerate(np.linspace(-size * largest, size * largest, res)):
        for j, y in enumerate(np.linspace(-size * largest, size * largest, res)):
            vector_pca = np.array([x, y, 0, 0])
            vector_n = pca.inverse_transform(np.expand_dims(vector_pca, axis=0)).squeeze()
            embs_n[res*i+j] = vector_n
    return embs_n, pca

# def grid_spanned_by_vectors(vectors):
#     res = 50
#     size = 3
#     embs_n = np.zeros((res*res, 4))
#     for l1 in np.linspace(-size, size, res):
#         for l2 in np.linspace(-size, size, res):

def get_points(model_path, points, vectors, **config):
    if type(points) == list:
        embs = points
        pca = None
    elif points == 'zero':
        embs = [torch.zeros(config['embed_dim'])]
        pca = None
    elif points == 'grid':
        embs, pca = pca_grid(model_path, vectors)
    elif points == 'avg':
        saved_embs = load_embs(model_path)
        embs = [saved_embs.mean(dim=0)]
        pca = None
    elif points == 'xygrid':
        embs = xygrid()
        pca = None
    return embs, pca

def eval_points(embeddings, model_path, device, **config):
    state_dict = torch.load(model_path, map_location='cpu')['model_state']

    val_model = get_model(**config)[0][0]
    val_model.load_state_dict(state_dict)
    val_model.to(device)

    ft_sites = get_ft_sites(**config)

    losses = np.zeros((len(embeddings), len(ft_sites)))
    accuracies = np.zeros((len(embeddings), len(ft_sites), 2))
    t1 = time.time()
    for i, emb in enumerate(embeddings):
        emb = torch.tensor(emb, device=device)
        val_model.embedding = nn.Parameter(emb)
        val_model.eval()
        l, a = validation(ft_sites, device, val_model, **config)
        losses[i] = l
        accuracies[i] = a
        if i % 100 == 0:
            t2 = time.time()
            print(f'{t2-t1:.2f}')
            t1 = time.time()
    classes = []
    for site in ft_sites:
        classes.append(np.unique(site['val_dl'].dataset.dataset.targets[site['val_dl'].dataset.indices]))

    return losses, accuracies, embeddings, classes

def validation(ft_sites, device, val_model, **config):
    losses = np.zeros(len(ft_sites))
    acc = np.zeros((len(ft_sites), 2))
    for i, site in enumerate(ft_sites):
        loader = site['val_dl']
        classes = np.unique(loader.dataset.dataset.targets[loader.dataset.indices])
        total = np.zeros(2)
        correct = np.zeros(2)
        for batch_tup in loader:
            batch, labels = batch_tup
            batch = batch.to(device=device, non_blocking=True).float().permute(0, 3, 1, 2)
            labels = labels.to(device=device, non_blocking=True).to(dtype=torch.long)

            batch, labels = transform_image(batch, labels, 'val', site['transform'], config['dataset'], config['model_name'])
            pred = val_model(batch)
            pred_label = torch.argmax(pred, dim=1)
            loss_fn = nn.CrossEntropyLoss()

            xe_loss = loss_fn(pred, labels)
            losses[i] += xe_loss
            for j, cl in enumerate(classes):
                cl_mask = labels == cl
                total[j] += cl_mask.sum()
                correct[j] += (pred_label == labels)[cl_mask].sum()
        acc[i] = correct / total
    return losses, acc

def get_ft_sites(degradation, site_number, data_part_seed, transform_gen_seed, tr_config, ft_site_number, cross_val_id, gl_seed, **config):

    trn_dl_list, val_dl_list = new_get_dl_lists(dataset=config['dataset'], batch_size=config['batch_size'], degradation=degradation, n_site=site_number, seed=data_part_seed, cross_val_id=cross_val_id, gl_seed=gl_seed)
    transform_list = get_test_transforms(site_number=site_number, seed=transform_gen_seed, degradation=degradation, device='cuda' if torch.cuda.is_available() else 'cpu', **tr_config)
    class_list = get_class_list(task='classification', site_number=site_number, class_number=18 if config['dataset'] == 'celeba' else None, class_seed=2, degradation=degradation)
    site_dict = [{'trn_dl': trn_dl_list[ndx],
                    'val_dl': val_dl_list[ndx],
                    'transform': transform_list[ndx],
                    'classes': class_list[ndx]}
                    for ndx in range(site_number)]

    ft_indices = get_ft_indices(site_number, ft_site_number, degradation)
    trn_site_dict = [site_dict[i] for i in range(len(site_dict)) if i not in ft_indices]
    ft_site_dict = [site_dict[i] for i in range(len(site_dict)) if i in ft_indices]

    return ft_site_dict