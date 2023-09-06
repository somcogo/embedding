import glob
import os

from training import LayerPersonalisationTrainingApp
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def finetune_and_evaluate(old_dir, old_comment, **kwargs):
    old_comment = kwargs['comment']
    search_string = os.path.join(old_dir, '*' + old_comment + '.state')
    path = glob.glob(search_string)[0]
    kwargs['comment'] = old_comment + '-finetune-siteindices4'
    kwargs['strategy'] = 'finetuning'
    app = LayerPersonalisationTrainingApp(**kwargs, finetuning=True,)
    as_is_validation = app.doValidation(2, app.val_dls)[1]
    normal_finetuning = app.main()

    if 'embed_dim' in kwargs:
        kwargs['comment'] = old_comment + '-onlyfc-siteindices4'
        kwargs['strategy'] = 'onlyfc'
        onlyfc_app = LayerPersonalisationTrainingApp(**kwargs, finetuning=True)
        only_fc_finetuning = onlyfc_app.main()

        kwargs['comment'] = old_comment + '-onlyemb-siteindices4'
        kwargs['strategy'] = 'onlyemb'
        onlyemb_app = LayerPersonalisationTrainingApp(**kwargs, finetuning=True)
        only_emb_finetuning = onlyemb_app.main()
    else:
        only_fc_finetuning, only_emb_finetuning = None, None

    return as_is_validation, normal_finetuning, only_fc_finetuning, only_emb_finetuning