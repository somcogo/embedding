import os
import glob

import torch

from main import ft_main
from config import get_finetuning_config

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.set_num_threads(8)

comment = 'fttest'
degradation = 'addgauss'
model = 'convnext'
dataset = 'cifar10'

cross_validate = False
# if not cross_validate:
#     cross_val_id = 0
# else:
cross_val_id = 0

config_fn = get_finetuning_config

if __name__ == '__main__':
    s1 = model[:2]
    s2 = dataset[:2]
    s3 = 'ji' if degradation == 'colorjitter' else 'ag' if degradation == 'addgauss' else 'cl'
    logdir = 'finalft/'+s1+s2+s3
    
    path = glob.glob(f'saved_models/final/{s1}{s2}{s3}/*vanilla*xval{cross_val_id}*')[0]
    state_dict = torch.load(path)['model_state']
    config = config_fn(logdir, comment, degradation, model, 'vanilla', dataset, cross_val_id=cross_val_id, fedbn=False)
    ft_main(state_dict=state_dict, **config)
    
    path = glob.glob(f'saved_models/final/{s1}{s2}{s3}/*vanilla*xval{cross_val_id}*')[0]
    state_dict = torch.load(path)['model_state']
    config = config_fn(logdir, comment, degradation, model, 'vanilla', dataset, cross_val_id=cross_val_id, fedbn=True)
    ft_main(state_dict=state_dict, **config)
    
    path = glob.glob(f'saved_models/final/{s1}{s2}{s3}/*embres2*xval{cross_val_id}*')[0]
    state_dict = torch.load(path)['model_state']
    config = config_fn(logdir, comment, degradation, model, 'embres2', dataset, cross_val_id=cross_val_id, fedbn=False)
    ft_main(state_dict=state_dict, **config)