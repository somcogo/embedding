import os

import numpy as np
import torch

from config import get_standard_config, get_exp_config
from main import main

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.set_num_threads(8)

logdir = 'resnetcoco'
comment = 'firsttry'
degradation = 'classsep'
model = 'resnet18'
model_type = 'vanilla'
dataset = 'minicoco'

cross_validate = False
# if not cross_validate:
#     cross_val_id = 0
# else:
cross_val_id = 0

config_fn = get_standard_config

if __name__ == '__main__':
    if cross_validate:
        results = {}
        for cross_val_id in range(5):
            config = config_fn(logdir, comment, degradation, model, model_type, dataset, cross_val_id=cross_val_id)
            results[cross_val_id] = main(**config)
        mean_res = {'training':sum([res['training'] for res in results.values()])/5}
        mean_res['fine_tuning'] = {key:sum([res['fine_tuning'][key] for res in results.values()])/5 for key in results[0]['fine_tuning'].keys()}

        save_path = '/home/hansel/developer/embedding/results/xvalidated'
        torch.save(mean_res, os.path.join(save_path, config['comment'].replace('xval4', 'xvalcombined')))
    else:
        config = config_fn(logdir, comment, degradation, model, model_type, dataset, cross_val_id=cross_val_id)
        main(**config)