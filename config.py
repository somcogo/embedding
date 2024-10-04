import numpy as np

def get_new_config(logdir, comment, site_number, degradation, comm_rounds, strategy, model_type, fed_prox, emb_dim, ft_site_number, cross_val_id, gl_seed, norm_layer, no_batch_running_stats, cl_per_site, dataset, fed_bn_emb_ft, feature_dim, trn_set_size, ft_emb_vec=None):
    dataset = 'digits' if degradation == ['digits'] else dataset
    batch_size = 64
    iterations = 50
    emb_dim = emb_dim if emb_dim is not None else 128
    emb_dim = None if model_type == 'vanilla' else emb_dim

    tr_config = {'var_add':(0.005, 1)}
    
    fflr = 1e-4 if model_type != 'vanilla' else None
    emb_lr = 1e-1 if model_type != 'vanilla' else None
    ft_emb_lr = 1e-2 if model_type != 'vanilla' else None

    if strategy == 'fedbntrn':
        ft_strategy = 'fedbn'
    elif strategy == 'noembed':
        ft_strategy = 'finetuning'
    elif strategy == 'embbn':
        ft_strategy = 'embbnft'
    elif strategy == 'pureemb':
        ft_strategy = 'onlyemb'
    elif strategy == 'all':
        ft_strategy = 'nomerge'

    deg_string = ''.join([name[:2] for name in degradation]) if type(degradation) == list else degradation
    feat_dim_string = feature_dim
    feature_dim = feature_dim * np.array([1, 2, 4, 8])

    config = {'logdir':logdir,
            'comment':f'{comment}-{strategy}-{model_type}-{deg_string}-s{str(site_number)}-fts{str(ft_site_number)}-lr1e-4-fflr{fflr}-elr{emb_lr}-ftelr{ft_emb_lr}-embdim-{emb_dim}-{dataset}-fedp-{str(fed_prox)}-xval{cross_val_id}-gls{gl_seed}-nl-{norm_layer}-fd-{feat_dim_string}-tss-{trn_set_size}-fbef-{fed_bn_emb_ft}',
            'model_name':'resnet18',
            'model_type':model_type,
            'degradation':degradation,
            'site_number':site_number,
            'embed_dim':emb_dim,
            'batch_size':batch_size,
            'cross_val_id':cross_val_id,
            'comm_rounds':comm_rounds,
            'lr':1e-4,
            'ffwrd_lr':fflr,
            'embedding_lr':emb_lr,
            'ft_emb_lr':ft_emb_lr,
            'weight_decay':1e-4,
            'optimizer_type':'newadam',
            'scheduler_mode':'cosine',
            'T_max':comm_rounds,
            'save_model':True,
            'strategy':strategy,
            'cifar':True,
            'data_part_seed':0,
            'transform_gen_seed':1,
            'dataset':dataset,
            'iterations':iterations,
            'tr_config':tr_config,
            'ft_strategy':ft_strategy,
            'fed_prox':fed_prox,
            'ft_site_number':ft_site_number,
            'gl_seed':gl_seed,
            'norm_layer':norm_layer,
            'no_batch_running_stats':no_batch_running_stats,
            'ft_emb_vec':ft_emb_vec,
            'cl_per_site':cl_per_site,
            'feature_dims':feature_dim,
            'trn_set_size':trn_set_size}
    
    return config