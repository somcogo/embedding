import glob
import os

import torch

from utils.data_loader import get_dl_lists
from models.model import ResNetWithEmbeddings
from loss_visualisation import visualise_loss, prepare_for_visualisation

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# trainings = ['s2alpha01', 's2alpha10', 's5alpha01', 's5alpha10']
# experiments = ['s1regular', 's2byclass', 's5byclass', 's10byclass', 's2dirichleta01', 's5dirichleta01', 's2dirichleta3', 's5dirichleta3', 's2dirichleta10', 's5dirichleta10']
# model = ResNetWithEmbeddings(num_classes=10)
model = ResNetWithEmbeddings(num_classes=10, layers=[2, 2, 2, 2], site_number=5, embed_dim=2, use_hypnns=True, version=1)
# for site in [2, 5, 8]:
for a in [0.1]:
    site = 5
    paths = glob.glob('saved_models/hypernetworks/2023_06_17-11_06_13_cifar10-resnet18emb-sgd-00001-e1500-b128-cosineT1500-s5-noembed-alpha01-hypnn1-embdim2.state')
    print(paths)
    model_path = paths[0]
    # state_dict, emb_vector_list, val_dls, x_min, x_max, x_step, y_min, y_max, y_step = prepare_for_visualisation(model_path)
    state_dict = torch.load(model_path)[0]['model_state']
    _, val_dls = get_dl_lists(dataset='cifar10', batch_size=1000, partition='dirichlet', alpha=0.1, n_site=2)
    x_min, x_max, x_step, y_min, y_max, y_step = -4, 4, 0.2, -4, 4, 0.2
    loss = visualise_loss(state_dict=state_dict, model=model, dl_list=val_dls, x_min=x_min, x_max=x_max, x_step=x_step, y_min=y_min, y_max=y_max, y_step=y_step, device=device)

    file_path = os.path.join(
        'plots',
        's2alpha01-s2alpha0.1-hypernn1edim2-b1000'
    )

    state = {
        'loss':loss,
        # 'emb_vector_list':emb_vector_list,
        'x_data':[x_max, x_min, x_step],
        'y_data':[y_max, y_min, y_step]
    }

    os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
    torch.save(state, file_path)