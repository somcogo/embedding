import os

import torch

from utils.data_loader import get_dl_lists
from models.model import ResNetWithEmbeddings
from loss_visualisation import visualise_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

trainings = ['s2alpha01', 's2alpha10', 's5alpha01', 's5alpha10']
experiments = ['s1regular', 's2byclass', 's5byclass', 's10byclass', 's2dirichleta01', 's5dirichleta01', 's2dirichleta3', 's5dirichleta3', 's2dirichleta10', 's5dirichleta10']
model = ResNetWithEmbeddings(num_classes=10)
for key1 in trainings:
    model_path = 'saved_models/embeddingtests/2023-05-25_cifar10-resnet34emb-sgd-0001-e500-b128-cosineT500-{}-noembed-{}.state'.format(key1[:2], key1[2:])
    state_dict = torch.load(model_path)[0]['model_state']
    del state_dict['embedding.weight']

    for site in [2, 5]:
        for a in [0.1, 3, 10]:
            _, val_dl_list = get_dl_lists(dataset='cifar10', partition='dirichlet', n_site=site, alpha=a, batch_size=500)
            loss = visualise_loss(state_dict=state_dict, model=model, dl_list=val_dl_list, x_min=-3.5, x_max=3.51, x_step=0.025, y_min=-3.5, y_max=3.51, y_step=0.025)

            file_path = os.path.join(
                'plots',
                '{}_s{}alpha{}'.format(key1, site, a)
            )
            os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
            torch.save(loss, file_path)
    for site in [2, 5, 10]:
        _, val_dl_list = get_dl_lists(dataset='cifar10', partition='by_class', n_site=site, batch_size=500)
        loss = visualise_loss(state_dict=state_dict, model=model, dl_list=val_dl_list, x_min=-3.5, x_max=3.51, x_step=0.025, y_min=-3.5, y_max=3.51, y_step=0.025)

        file_path = os.path.join(
            'plots',
            '{}_s{}byclass'.format(key1, site)
        )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        torch.save(loss, file_path)