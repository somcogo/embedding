import os
import glob

import torch

from main import ft_main
from config import get_finetuning_config

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.set_num_threads(8)

logdir = 'resnetcifar10bn4/finetuning'
comment = 'fttest'
degradation = 'classsep'
model = 'resnet18'
dataset = 'cifar10'
model_type = 'embbn4'
ft_strategy = 'onlyemb'

cross_validate = False
# if not cross_validate:
#     cross_val_id = 0
# else:
cross_val_id = 0

config_fn = get_finetuning_config

if __name__ == '__main__':
    
    path = glob.glob(f'saved_models/resnetcifar10bn4/*{model}*{model_type}*{degradation}*fflr-0.0001*')[0]
    state_dict = torch.load(path)['model_state']
    config = config_fn(logdir, comment, degradation, model, model_type, dataset, cross_val_id=cross_val_id, strategy=ft_strategy)
    ft_main(state_dict=state_dict, **config)