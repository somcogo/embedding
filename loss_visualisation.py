import os

import numpy as np
import torch

from utils.data_loader import get_dl_lists
from utils.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = 'saved_models/embeddingtests/2023-05-25_cifar10-resnet34emb-sgd-0001-e500-b128-cosineT500-s2-noembed-alpha01.state'

# model = ResNetWithEmbeddings(num_classes=10)
# state_dict = torch.load(model_path)[0]['model_state']
# del state_dict['embedding.weight']

# _, val_dl_list = get_dl_lists(dataset='cifar10', partition='dirichlet', n_site=5, alpha=0.1, batch_size=500)

def visualise_loss(state_dict, model, dl_list, x_min, x_max, x_step, y_min, y_max, y_step, device):
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    x_values = np.arange(x_min, x_max, x_step)
    y_values = np.arange(y_min, y_max, y_step)
    loss_values = np.zeros((len(dl_list), len(x_values), len(y_values)))

    with torch.no_grad():
        model.eval()
        for site_id, dl in enumerate(dl_list):
            log.info(site_id)
            for (batch, labels) in dl:
                batch = batch.to(device=device, non_blocking=True).float().permute(0, 3, 1, 2)
                labels = labels.to(device=device, non_blocking=True).to(dtype=torch.long)
                for x_ndx, x in enumerate(x_values):
                    for y_ndx, y in enumerate(y_values):
                        model.embedding.weight[0] = torch.nn.Parameter(torch.tensor([x, y], dtype=torch.float, device=device))
                        pred = model(batch, torch.tensor(0, device=device, dtype=torch.int))
                        loss_values[site_id, x_ndx, y_ndx] += loss_fn(pred, labels).sum()
                break
    return loss_values

def prepare_for_visualisation(model_path):
    loaded_dict = torch.load(model_path)
    state_dict = loaded_dict[0]['model_state']
    del state_dict['embedding.weight']
    emb_vector_list = torch.zeros((len(loaded_dict.keys())-2, loaded_dict[0]['emb_vector'].shape[0]))
    for ndx in range(emb_vector_list.shape[0]):
        emb_vector_list[ndx] = loaded_dict[ndx]['emb_vector']
    trn_idx_map = {ndx:loaded_dict[ndx]['trn_indices'] for ndx in range(len(emb_vector_list))}
    val_idx_map = {ndx:loaded_dict[ndx]['val_indices'] for ndx in range(len(emb_vector_list))}
    _, val_dls = get_dl_lists(dataset='cifar10', batch_size=500, partition='given',net_dataidx_map_train=trn_idx_map, net_dataidx_map_test=val_idx_map)

    center_point = emb_vector_list.mean(dim=0)
    axis_mins = torch.amin(emb_vector_list, dim=0)
    axis_maxs = torch.amax(emb_vector_list, dim=0)
    axis_abs = axis_maxs - axis_mins
    axis_abs = torch.max(axis_abs, torch.ones(axis_abs.shape))
    x_max = center_point[0] + axis_abs[0]
    x_min = center_point[0] - axis_abs[0]
    x_step = (x_max-x_min) / 50
    y_max = center_point[1] + axis_abs[1]
    y_min = center_point[1] - axis_abs[1]
    y_step = (y_max-y_min) / 50
    return state_dict, emb_vector_list, val_dls, x_min, x_max, x_step, y_min, y_max, y_step


