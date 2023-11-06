import glob
import os

import numpy as np
import torch

from training import EmbeddingTraining

def finetune_and_evaluate(old_dir, **kwargs):
    old_comment = kwargs['comment']
    search_string = os.path.join('saved_models', old_dir, '*' + old_comment + '.state')
    print(search_string)
    path = glob.glob(search_string)[0]
    kwargs['comment'] = old_comment + '-finetune-siteindices4'
    kwargs['strategy'] = 'finetuning'
    app = EmbeddingTraining(**kwargs, finetuning=True, model_path=path)
    as_is_validation = app.doValidation(2, app.val_dls)[1]
    normal_finetuning = app.main()

    if kwargs['embed_dim'] is not None:
        if kwargs['fc_residual']:
            kwargs['comment'] = old_comment + '-onlyfc-siteindices4'
            kwargs['strategy'] = 'onlyfc'
            onlyfc_app = EmbeddingTraining(**kwargs, finetuning=True, model_path=path)
            only_fc_finetuning = onlyfc_app.main()
        else:
            only_fc_finetuning = None

        kwargs['comment'] = old_comment + '-onlyemb-siteindices4'
        kwargs['strategy'] = 'onlyemb'
        onlyemb_app = EmbeddingTraining(**kwargs, finetuning=True, model_path=path)
        only_emb_finetuning = onlyemb_app.main()
    else:
        only_fc_finetuning, only_emb_finetuning = None, None

    os.makedirs(os.path.join('data/tables', old_dir), exist_ok=True)
    torch.save([as_is_validation, normal_finetuning, only_fc_finetuning, only_emb_finetuning], os.path.join('data/tables', old_dir, old_comment))
    os.makedirs(os.path.join('data/tables', 'dict', old_dir), exist_ok=True)
    torch.save({'as_is':as_is_validation, 'normal':normal_finetuning, 'onlyfc':only_fc_finetuning, 'onlyemb':only_emb_finetuning}, os.path.join('data/tables', 'dict', old_dir, old_comment))

    return as_is_validation, normal_finetuning, only_fc_finetuning, only_emb_finetuning


def input_variation(perm_seed, trn_site_number=4, finetune_epochs=100, resnet18_too=True, **inputs):
    kwargs = {
        'epochs':500,
        'logdir':'mnist/inputvariation',
        'comment':'resnet18emb-s5-alpha1e7-embdim32-pseed220',
        'dataset':'mnist',
        'site_number':5,
        'model_name':'resnet18emb',
        'save_model':True,
        'partition':'dirichlet',
        'alpha':1e7,
        'strategy':'noembed',
        'embed_dim':32,
        'k_fold_val_id':None,
        'seed':0,
        'site_indices':[0, 1, 2, 3],
        'input_perturbation':True,
        'use_hdf5':True,
        'embedding_lr':None,
        'conv1_residual':True,
        'fc_residual':True
        }
    
    for key in inputs.keys():
        kwargs[key] = inputs[key]
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(kwargs['site_number'])
    kwargs['site_indices'] = perm[:trn_site_number]
    print('***Training for permutation {}***'.format(perm))

    exp = EmbeddingTraining(**kwargs)
    exp.main()

    old_dir = kwargs['logdir']
    old_epochs = kwargs['epochs']
    kwargs['logdir'] = os.path.join(old_dir, 'finetuning')
    kwargs['epochs'] = finetune_epochs
    kwargs['site_indices'] = perm[trn_site_number:]
    results = finetune_and_evaluate(old_dir=old_dir, **kwargs)

    if resnet18_too:
        kwargs['comment'] = kwargs['comment'].replace(kwargs['model_name'], 'resnet18')
        kwargs['model_name'] = 'resnet18'
        kwargs['logdir'] = old_dir
        kwargs['epochs'] = old_epochs
        kwargs['site_indices'] = perm[:trn_site_number]
        kwargs['embed_dim'] = None

        exp = EmbeddingTraining(**kwargs)
        exp.main()

        kwargs['logdir'] = os.path.join(old_dir, 'finetuning')
        kwargs['epochs'] = finetune_epochs
        kwargs['site_indices'] = perm[trn_site_number:]
        results = finetune_and_evaluate(old_dir=old_dir, **kwargs)

def by_class_seperation(ndx, finetuning_epochs, **inputs):
    kwargs = {
        'epochs':500,
        'logdir':'mnist/classseperation',
        'comment':'resnet18emb-s5-byclass-embdim32-gn-seed0',
        'dataset':'mnist',
        'site_number':5,
        'model_name':'resnet18emb',
        'save_model':True,
        'partition':'by_class',
        'strategy':'noembed',
        'embed_dim':32,
        'k_fold_val_id':None,
        'seed':0,
        'site_indices':[0, 1, 2, 3],
        'use_hdf5':True,
        'conv1_residual':True,
        'fc_residual':True
        }
    
    for key in inputs.keys():
        kwargs[key] = inputs[key]

    exp = EmbeddingTraining(**kwargs)
    exp.main()

    old_dir = kwargs['logdir']
    old_epochs = kwargs['epochs']
    kwargs['logdir'] = os.path.join(old_dir, 'finetuning')
    kwargs['epochs'] = finetuning_epochs
    kwargs['site_indices'] = [4]
    results = finetune_and_evaluate(old_dir=old_dir, **kwargs)

    kwargs['comment'] = kwargs['comment'].replace(kwargs['model_name'], 'resnet18')
    kwargs['model_name'] = 'resnet18'
    kwargs['logdir'] = old_dir
    kwargs['epochs'] = old_epochs
    kwargs['site_indices'] = [0, 1, 2, 3]
    kwargs['embed_dim'] = None

    exp = EmbeddingTraining(**kwargs)
    exp.main()

    kwargs['logdir'] = os.path.join(old_dir, 'finetuning')
    kwargs['epochs'] = finetuning_epochs
    kwargs['site_indices'] = [4]
    results = finetune_and_evaluate(old_dir=old_dir, **kwargs)