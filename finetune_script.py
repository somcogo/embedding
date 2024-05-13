import os

import torch

from main import ft_main
from config import get_finetuning_config

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.set_num_threads(8)

logdir = 'convcoco'
comment = 'lr-4'
degradation = 'classsep'
model = 'convnext'
model_type = 'vanilla'
dataset = 'minicoco'

model_path = ''
state_dict = torch.load(model_path)['model_state']

cross_validate = False
# if not cross_validate:
#     cross_val_id = 0
# else:
cross_val_id = 0

config_fn = get_finetuning_config

if __name__ == '__main__':
    config = config_fn(logdir, comment, degradation, model, model_type, dataset, cross_val_id=cross_val_id)
    ft_main(**config)