import glob
import os

from training import EmbeddingTraining
import torch

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

    if 'embed_dim' in kwargs:
        kwargs['comment'] = old_comment + '-onlyfc-siteindices4'
        kwargs['strategy'] = 'onlyfc'
        onlyfc_app = EmbeddingTraining(**kwargs, finetuning=True, model_path=path)
        only_fc_finetuning = onlyfc_app.main()

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