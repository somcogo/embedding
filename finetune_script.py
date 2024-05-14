import os

import torch

from main import ft_main
from config import get_finetuning_config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(8)

logdir = 'resnetcoco'
comment = 'finetune'
degradation = 'addgauss'
model = 'resnet18'
model_type = 'embres2'
dataset = 'minicoco'

# model_path = 'saved_models/convcelebav4/2024_05_12-00_16_47-long-segmentation-convnext-embres2-colorjitter-s5-ts2-edim128-b32-commr800-ftcr100-iter50-lr0.004-emblr1e-3-adamw-warmcos-ftcosine-wd0.05-celeba-augold-alpha10000000.0-fdim90-ls0.1-xval0.state'
model_path = 'saved_models/resnetcoco/2024_05_13-08_47_48-cocotest-segmentation-resnet18-embres2-addgauss-s5-ts2-edim128-b32-commr400-ftcr100-iter50-lr1e-4-fflrNone-emblr1e-3-minicoco-alpha10000000.0-fdim62-xval0.state'
state_dict = torch.load(model_path)['model_state']

cross_validate = False
# if not cross_validate:
#     cross_val_id = 0
# else:
cross_val_id = 0

config_fn = get_finetuning_config

if __name__ == '__main__':
    config = config_fn(logdir, comment, degradation, model, model_type, dataset, cross_val_id=cross_val_id)
    ft_main(state_dict=state_dict, **config)