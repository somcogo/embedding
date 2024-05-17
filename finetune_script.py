import os

import torch

from main import ft_main
from config import get_finetuning_config

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_num_threads(8)

logdir = 'resnetcifar10'
comment = 'fedbn'
degradation = 'addgauss'
model = 'resnet18'
model_type = 'vanilla'
dataset = 'cifar10'

# model_path = 'saved_models/convcelebav4/2024_05_12-00_16_47-long-segmentation-convnext-embres2-colorjitter-s5-ts2-edim128-b32-commr800-ftcr100-iter50-lr0.004-emblr1e-3-adamw-warmcos-ftcosine-wd0.05-celeba-augold-alpha10000000.0-fdim90-ls0.1-xval0.state'
# model_path = 'saved_models/resnetcoco/2024_05_13-08_47_48-cocotest-segmentation-resnet18-embres2-addgauss-s5-ts2-edim128-b32-commr400-ftcr100-iter50-lr1e-4-fflrNone-emblr1e-3-minicoco-alpha10000000.0-fdim62-xval0.state'
# model_path = 'saved_models/resnetcoco/2024_05_12-00_09_18-cocotest-segmentation-resnet18-embres2-classsep-s5-ts2-edim128-b32-commr400-ftcr100-iter50-lr1e-4-fflrNone-emblr1e-3-minicoco-alpha0.1-fdim62-xval0.state'
# model_path = 'saved_models/resnetcoco/2024_05_12-00_08_48-cocotest-segmentation-resnet18-embres2-colorjitter-s5-ts2-edim128-b32-commr400-ftcr100-iter50-lr1e-4-fflrNone-emblr1e-3-minicoco-alpha10000000.0-fdim62-xval0.state'
# model_path = 'saved_models/resnetcifar10/2024_04_30-20_45_55-firsttry-classification-resnet18-vanilla-colorjitter-s5-ts2-edimNone-b64-commr3200-ftcr800-iter50-lr1e-4-fflrNone-emblr1e-3-cifar10-alpha10000000.0-fdim64-xval4.state'
# model_path = 'saved_models/resnetcifar10/2024_04_29-17_49_30-firsttry-classification-resnet18-vanilla-classsep-s5-ts2-edimNone-b64-commr3200-ftcr800-iter50-lr1e-4-fflrNone-emblr1e-3-cifar10-alphaclasssep-fdim64-xval4.state'
model_path = 'saved_models/resnetcifar10/2024_04_29-16_18_50-firsttry-classification-resnet18-vanilla-addgauss-s5-ts2-edimNone-b64-commr3200-ftcr800-iter50-lr1e-4-fflrNone-emblr1e-3-cifar10-alpha10000000.0-fdim64-xval3.state'




state_dict = torch.load(model_path)['model_state']

cross_validate = False
# if not cross_validate:
#     cross_val_id = 0
# else:
cross_val_id = 3

config_fn = get_finetuning_config

if __name__ == '__main__':
    config = config_fn(logdir, comment, degradation, model, model_type, dataset, cross_val_id=cross_val_id)
    ft_main(state_dict=state_dict, **config)