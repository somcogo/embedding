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

def pca_grid(model_path):
    saved_embs = load_embs(model_path)
    pca = PCA(n_components=saved_embs.shape[1])
    pca.fit(saved_embs)
    tr_xy = pca.transform(saved_embs)[:2]
    diam_x, diam_y = np.linalg.norm(tr_xy, axis=1)
    center = tr_xy.mean(axis=1)
    embs = []
    for x in np.linspace(-18 * diam_x + center[0], 2 * diam_x + center[0], 100):
        for y in np.linspace(-18 * diam_y + center[1], 2 * diam_y + center[1], 100):
            embs.append(torch.from_numpy(pca.inverse_transform(np.array([x, y, 0, 0]))))    
    return embs

def get_points(model_path, points, **config):
    if type(points) == list:
        embs = points
    elif points == 'zero':
        embs = [torch.zeros(config['embed_dim'])]
    elif points == 'grid':
        embs = pca_grid(model_path)
    elif points == 'avg':
        saved_embs = load_embs(model_path)
        embs = [saved_embs.mean(dim=0)]
    elif points == 'xygrid':
        embs = xygrid()
    return embs

def eval_points(embeddings, model_path, device, **config):
    state_dict = torch.load(model_path, map_location='cpu')['model_state']

    val_model = get_model(**config)[0][0]
    val_model.load_state_dict(state_dict)

    losses = []
    t1 = time.time()
    for i, emb in enumerate(embeddings):
        val_model.embedding = nn.Parameter(emb)
        losses.append(eval_model(val_model=val_model, device=device, **config))
        if i % 100 == 0:
            t2 = time.time()
            print(f'{t2-t1:.2f}')
            t1 = time.time()

    return losses, embeddings


def eval_model(val_model, device, degradation, site_number, data_part_seed, transform_gen_seed, tr_config, ft_site_number, cross_val_id, gl_seed, **config):
    val_model.eval()
    val_model.to(device)

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

    losses = np.zeros((len(ft_site_dict)), dtype=float)
    for i, site in enumerate(ft_site_dict):
        loader = site['val_dl']
        for batch_tup in loader:
            batch, labels = batch_tup
            batch = batch.to(device=device, non_blocking=True).float()
            labels = labels.to(device=device, non_blocking=True).to(dtype=torch.long)
            if config['dataset'] in ['cifar10', 'cifar100']:
                batch = batch.permute(0, 3, 1, 2)
            if config['dataset'] in ['celeba']:
                batch = batch.permute(0, 3, 1, 2)
                labels = create_mask_from_onehot(labels, site['classes'])

            batch, labels = transform_image(batch, labels, 'val', site['transform'], config['dataset'], config['model_name'])
            pred = val_model(batch)
            loss_fn = nn.CrossEntropyLoss()

            xe_loss = loss_fn(pred, labels)
            losses[i] += xe_loss
    
    return losses