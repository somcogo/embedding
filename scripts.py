import glob
import os

import numpy as np
import torch

from training import LayerPersonalisationTrainingApp

def finetune_and_evaluate(old_dir, **kwargs):
    old_comment = kwargs['comment']
    search_string = os.path.join('saved_models', old_dir, '*' + old_comment + '.state')
    print(search_string)
    path = glob.glob(search_string)[0]
    kwargs['comment'] = old_comment + '-finetune-siteindices4'
    kwargs['strategy'] = 'finetuning'
    app = LayerPersonalisationTrainingApp(**kwargs, finetuning=True, model_path=path)
    as_is_validation = app.doValidation(2, app.val_dls)[1]
    normal_finetuning = app.main()

    if 'embed_dim' in kwargs:
        kwargs['comment'] = old_comment + '-onlyfc-siteindices4'
        kwargs['strategy'] = 'onlyfc'
        onlyfc_app = LayerPersonalisationTrainingApp(**kwargs, finetuning=True, model_path=path)
        only_fc_finetuning = onlyfc_app.main()

        kwargs['comment'] = old_comment + '-onlyemb-siteindices4'
        kwargs['strategy'] = 'onlyemb'
        onlyemb_app = LayerPersonalisationTrainingApp(**kwargs, finetuning=True, model_path=path)
        only_emb_finetuning = onlyemb_app.main()
    else:
        only_fc_finetuning, only_emb_finetuning = None, None

    os.makedirs(os.path.join('data/tables', old_dir), exist_ok=True)
    torch.save([as_is_validation, normal_finetuning, only_fc_finetuning, only_emb_finetuning], os.path.join('data/tables', old_dir, old_comment))
    os.makedirs(os.path.join('data/tables', 'dict', old_dir), exist_ok=True)
    torch.save({'as_is':as_is_validation, 'normal':normal_finetuning, 'onlyfc':only_fc_finetuning, 'onlyemb':only_emb_finetuning}, os.path.join('data/tables', 'dict', old_dir, old_comment))

    return as_is_validation, normal_finetuning, only_fc_finetuning, only_emb_finetuning


kwargs = {
    'dataset':kwargs['dataset'],
    'model_name':kwargs['model_name'],
    'site_number':5,
    'partition':'dirichlet',
    'alpha':1e7,
    'seed':0,
    'comment':'{}-s5-alpha1e7-embdim32-seed{}'.format(model_name, seed),
    'logdir':'{}/inputvariation'.format(dataset),
    'epochs':500,
    'site_indices':perm[:4],
    'save_model':True,
    'embed_dim':32,
    'use_hdf5':True
    }
def compare_emb_and_finetune(**kwargs):
    dataset = 'cifar10'
    for seed in range(51, 56):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(5)
        print('***Training for permutation {}***'.format(perm))

        exp = LayerPersonalisationTrainingApp(**kwargs)
        exp.main()

        kwargs['logdir'] = '{}/inputvariation/finetuning'.format(dataset)
        kwargs['epochs'] = 100
        kwargs['site_indices'] = perm[4:]
        results = finetune_and_evaluate(old_dir='{}/inputvariation'.format(dataset), **kwargs)

        kwargs['model_name'] = 'resnet18'
        kwargs['comment'] = 'resnet18-e1000-s5-noembed-byclass-seed{}-v5'.format(seed)
        kwargs['logdir'] = '{}/inputvariation'.format(dataset)
        kwargs['epochs'] = 500
        kwargs['site_indices'] = perm[:4]
        del kwargs['embed_dim']

        exp = LayerPersonalisationTrainingApp(**kwargs)
        exp.main()

        kwargs['logdir'] = '{}/inputvariation/finetuning'.format(dataset)
        kwargs['epochs'] = 100
        kwargs['site_indices'] = perm[4:]
        results = finetune_and_evaluate(old_dir='{}/inputvariation'.format(dataset), **kwargs)