import numpy as np

def get_config(logdir, comment, emb_dim, task, site_number, trn_site_number, degradation, model_type, comm_rounds, ft_comm_rounds, feature_dims=None, cross_val_id=None):
    if task == 'classification':
        batch_size = 64
        dataset = 'imagenet'
        if site_number > 1:
            iterations = 50
        else:
            iterations = None
    elif task == 'segmentation':
        batch_size = 32
        dataset = 'celeba'
        if site_number > 1:
            iterations = 50
        else:
            iterations = None
    
    if degradation == 'classsep':
        partition = 'by_class'
        alpha = None
    else:
        partition = 'dirichlet'
        alpha = 1e7
    alpha_str = str(alpha) if partition == 'dirichlet' else 'classsep'

    tr_config = {'var_add':(0.005, 1),
                'alpha':(1.2, 1.5),
                'var_mul':(0.01, 0.5),
                'patch_size':3,
                'swap_count':1}
    
    if 'emb' in model_type:
        if cross_val_id is None and task == 'classification':
            ft_strategies = ['finetuning', 'onlyfc', 'onlyemb', 'fffinetuning']
        else:
            ft_strategies = ['fffinetuning']
        feature_dims = [62, 124, 248, 496] if feature_dims is None else feature_dims
    else:
        ft_strategies = ['finetuning']
        feature_dims = None
    fdim_str = str(64 if feature_dims is None else feature_dims[0])

    config = {'logdir':logdir,
            'comment':f'{comment}-{task}-resnet18-{model_type}-cifar-{degradation}-s{str(site_number)}-ts{str(trn_site_number)}-edim{str(emb_dim)}-b{str(batch_size)}-commr{str(comm_rounds)}-ftcr{str(ft_comm_rounds)}-iter{str(iterations)}-lr1e-4-fflrNone-emblr1e-3-{dataset}-alpha{alpha_str}-fdim{fdim_str}-xval{cross_val_id}',
            'task':task,
            'model_name':'resnet18',
            'model_type':model_type,
            'degradation':degradation,
            'site_number':site_number,
            'trn_site_number':trn_site_number,
            'embedding_dim':emb_dim,
            'batch_size':batch_size,
            'cross_val_id':cross_val_id,
            'comm_rounds':comm_rounds,
            'ft_comm_rounds':ft_comm_rounds,
            'lr':1e-4,
            'ff_lr':None,
            'emb_lr':1e-3,
            'weight_decay':1e-4,
            'optimizer_type':'newadam',
            'scheduler_mode':'cosine',
            'T_max':comm_rounds,
            'save_model':True,
            'strategy':'noembed',
            'ft_strategies':ft_strategies,
            'cifar':True,
            'data_part_seed':0,
            'transform_gen_seed':1,
            'dataset':dataset,
            'alpha':alpha,
            'iterations':iterations,
            'tr_config':tr_config,
            'partition':partition,
            'feature_dims':feature_dims}
    
    return config

def get_standard_config(logdir, comment, degradation, model, model_type, dataset, cross_val_id=None):
    if dataset in ['cifar10', 'imagenet']:
        task = 'classification'
        batch_size = 64 if model == 'resnet18' else 256
        comm_rounds = 3200
        ft_comm_rounds = 800
    elif dataset in ['celeba']:
        task = 'segmentation'
        batch_size = 32
        comm_rounds = 400
        ft_comm_rounds = 100
    
    if degradation == 'classsep':
        partition = 'by_class'
        alpha = None
    else:
        partition = 'dirichlet'
        alpha = 1e7
    alpha_str = str(alpha) if partition == 'dirichlet' else 'classsep'

    tr_config = {'var_add':(0.005, 1),
                'alpha':(1.2, 1.5),
                'var_mul':(0.01, 0.5),
                'patch_size':3,
                'swap_count':1}
    
    if 'emb' in model_type:
        ft_strategies = ['fffinetuning']
        emb_dim = 128
        if model == 'resnet18':
            feature_dims = 62 * np.array([1, 2, 4, 8])
        elif model == 'convnext':
            feature_dims = 90 * np.array([1, 2, 4, 8])
    else:
        ft_strategies = ['finetuning']
        emb_dim = None
        if model == 'resnet18':
            feature_dims = 64 * np.array([1, 2, 4, 8])
        elif model == 'convnext':
            feature_dims = 96 * np.array([1, 2, 4, 8])
    fdim_str = str(64 if feature_dims is None else feature_dims[0])

    config = {'logdir':logdir,
            'comment':f'{comment}-{task}-{model}-{model_type}-{degradation}-s5-ts2-edim{str(emb_dim)}-b{str(batch_size)}-commr{str(comm_rounds)}-ftcr{str(ft_comm_rounds)}-iter50-lr1e-4-fflrNone-emblr1e-3-{dataset}-alpha{alpha_str}-fdim{fdim_str}-xval{cross_val_id}',
            'task':task,
            'model_name':model,
            'model_type':model_type,
            'degradation':degradation,
            'site_number':5,
            'trn_site_number':2,
            'embedding_dim':emb_dim,
            'batch_size':batch_size,
            'cross_val_id':cross_val_id,
            'comm_rounds':comm_rounds,
            'ft_comm_rounds':ft_comm_rounds,
            'lr':1e-4,
            'ff_lr':None,
            'emb_lr':1e-3,
            'weight_decay':1e-4,
            'optimizer_type':'newadam',
            'scheduler_mode':'cosine',
            'T_max':comm_rounds,
            'save_model':True,
            'strategy':'noembed',
            'ft_strategies':ft_strategies,
            'cifar':True,
            'data_part_seed':0,
            'transform_gen_seed':1,
            'dataset':dataset,
            'alpha':alpha,
            'iterations':50,
            'tr_config':tr_config,
            'partition':partition,
            'feature_dims':feature_dims}
    
    return config
