import glob
import os

import torch

from utils.data_loader import get_dl_lists
from models.model import ResNetWithEmbeddings
from loss_visualisation import visualise_loss, prepare_for_visualisation

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# trainings = ['s2alpha01', 's2alpha10', 's5alpha01', 's5alpha10']
# experiments = ['s1regular', 's2byclass', 's5byclass', 's10byclass', 's2dirichleta01', 's5dirichleta01', 's2dirichleta3', 's5dirichleta3', 's2dirichleta10', 's5dirichleta10']
model = ResNetWithEmbeddings(num_classes=10)
# for site in [2, 5, 8]:
for a in [0.1, 3, 10]:
    site = 8
    paths = glob.glob('saved_models/finetuning/*cifar10-resnet18emb-lr0001-b128-s2alpha01-s{}dirichlet{}-sgd-run2.state'.format(site, a))
    print(paths)
    model_path = paths[0]
    state_dict, emb_vector_list, val_dls, x_max, x_min, x_step, y_max, y_min, y_step = prepare_for_visualisation(model_path)

    loss = visualise_loss(state_dict=state_dict, model=model, dl_list=val_dls, x_min=x_min, x_max=x_max, x_step=x_step, y_min=y_min, y_max=y_max, y_step=y_step, device=device)

    file_path = os.path.join(
        'plots',
        's2alpha01-s{}alpha{}-v3'.format(site, a)
    )

    state = {
        'loss':loss,
        'emb_vector_list':emb_vector_list,
        'x_data':[x_max, x_min, x_step],
        'y_data':[y_max, y_min, y_step]
    }

    os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
    torch.save(state, file_path)