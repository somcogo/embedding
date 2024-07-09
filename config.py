import numpy as np

def get_new_config(logdir, comment, site_number, degradation, comm_rounds, strategy, model_type, fed_prox, prox_map, emb_dim, ft_site_number, cross_val_id, gl_seed):
    dataset = 'cifar10'
    task = 'classification'
    partition = 'dirichlet'
    batch_size = 64
    iterations = 50
    alpha = 1e7
    emb_dim = emb_dim if emb_dim is not None else 128
    emb_dim = None if model_type == 'vanilla' else emb_dim

    tr_config = {'var_add':(0.005, 1),
                'alpha':(1.2, 1.5),
                'var_mul':(0.01, 0.5),
                'patch_size':3,
                'swap_count':1}
    
    fflr = 1e-4 if model_type != 'vanilla' else None
    emb_lr = 1e-1 if model_type != 'vanilla' else None
    ft_emb_lr = 1e-3 if model_type != 'vainlla' else None

    if strategy == 'fedbntrn':
        ft_strategy = 'fedbn'
    elif strategy == 'noembed':
        ft_strategy = 'finetuning'
    elif strategy == 'embbn':
        ft_strategy = 'embbnft'
    elif strategy == 'pureemb':
        ft_strategy = 'onlyemb'

    config = {'logdir':logdir,
            'comment':f'{comment}-{strategy}-{task}-resnet18-{model_type}-{degradation}-s{str(site_number)}-fts{str(ft_site_number)}-b{str(batch_size)}-commr{str(comm_rounds)}-iter{str(iterations)}-lr1e-4-fflr{fflr}-elr{emb_lr}-ftelr{ft_emb_lr}-embdim-{emb_dim}-{dataset}-fedp-{str(fed_prox)}-proxm-{prox_map}-xval{cross_val_id}-gls{gl_seed}',
            'task':task,
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
            'alpha':alpha,
            'iterations':iterations,
            'tr_config':tr_config,
            'partition':partition,
            'ft_strategy':ft_strategy,
            'fed_prox':fed_prox,
            'proximal_map':prox_map,
            'ft_site_number':ft_site_number,
            'gl_seed':gl_seed}
    
    return config


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
        batch_size = 64 if model == 'resnet18' else 64
        comm_rounds = 3200
        ft_comm_rounds = 800
    elif dataset in ['celeba', 'minicoco']:
        task = 'segmentation'
        batch_size = 32
        comm_rounds = 400 if model == 'resnet18' else 100
        ft_comm_rounds = 100 if model == 'resnet18' else 50
    
    if degradation == 'classsep':
        if dataset == 'minicoco':
            partition = 'dirichlet'
            alpha = 0.1
        else:
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

def get_exp_config(logdir, comment, degradation, model, model_type, dataset, cross_val_id=None, strategy=None, ft_strategy=None):
    if dataset in ['cifar10', 'imagenet']:
        task = 'classification'
        batch_size = 64 if model == 'resnet18' else 64
        comm_rounds = 3200
        ft_comm_rounds = 400
    elif dataset in ['celeba', 'minicoco']:
        task = 'segmentation'
        batch_size = 32
        comm_rounds = 400
        ft_comm_rounds = 100
    
    if degradation == 'classsep':
        if dataset == 'minicoco':
            partition = 'dirichlet'
            alpha = 0.1
        else:
            partition = 'by_class'
            alpha = None
    else:
        partition = 'dirichlet'
        alpha = 1e7

    tr_config = {'var_add':(0.005, 1),
                'alpha':(1.2, 1.5),
                'var_mul':(0.01, 0.5),
                'patch_size':3,
                'swap_count':1}
    
    if 'emb' in model_type:
        # ft_strategies = []
        emb_dim = 128
        if model == 'resnet18':
            feature_dims = 62 * np.array([1, 2, 4, 8])
        elif model in ['convnext', 'convnextog', 'swinv2']:
            feature_dims = 90 * np.array([1, 2, 4, 8])
    else:
        # ft_strategies = []
        emb_dim = None
        if model == 'resnet18':
            feature_dims = 64 * np.array([1, 2, 4, 8])
        elif model in ['convnext', 'convnextog', 'swinv2']:
            feature_dims = 96 * np.array([1, 2, 4, 8])
    fdim_str = str(64 if feature_dims is None else feature_dims[0])

    iterations = 50 if model == 'resnet18' else 50
    site_number = 5
    trn_site_number = 2
    # lr = 1e-4 if dataset == 'minicoco' else 4e-3
    # emb_lr = 1e-3
    # ft_lr = 1e-5 if dataset == 'minicoco' else lr
    # ft_emb_lr = 1e-5 if task == 'segmentation' else 1e-3
    lr = 1e-4
    ff_lr = 1e-2
    emb_lr = 1e-1
    ft_lr = 1e-4
    ft_ff_lr = 1e-2
    ft_emb_lr = 1e-1
    weight_decay = 1e-5
    label_smoothing = 0. if model == 'resnet18' else 0.1

    optimizer = 'newadam' if model == 'resnet18'else 'adamw'
    scheduler = 'cosine' if model == 'resnet18'else 'warmcos'
    ft_scheduler = 'cosine'

    trn_logging = True
    aug = 'old' if trn_logging else 'new'

    strategy = strategy if strategy is not None else 'noembed'
    ft_strategies = ft_strategy if ft_strategy is not None else 'finetuning'

    config = {'logdir':logdir,
            'comment':f'{comment}-{model}-{model_type}-{degradation}-{strategy}-s{str(site_number)}-ts{str(trn_site_number)}-edim{str(emb_dim)}-b{str(batch_size)}-commr{str(comm_rounds)}-ftcr{str(ft_comm_rounds)}-lr{str(lr)}-fflr-{str(ff_lr)}-emblr{str(emb_lr)}-{optimizer}-{scheduler}-ft{ft_scheduler}-wd{str(weight_decay)}-{dataset}-fdim{fdim_str}-ls{str(label_smoothing)}-xval{cross_val_id}',
            'task':task,
            'model_name':model,
            'model_type':model_type,
            'degradation':degradation,
            'site_number':site_number,
            'trn_site_number':trn_site_number,
            'embedding_dim':emb_dim,
            'batch_size':batch_size,
            'cross_val_id':cross_val_id,
            'comm_rounds':comm_rounds,
            'ft_comm_rounds':ft_comm_rounds,
            'lr':lr,
            'ff_lr':ff_lr,
            'emb_lr':emb_lr,
            'ft_lr':ft_lr,
            'ft_ff_lr':ft_ff_lr,
            'ft_emb_lr':ft_emb_lr,
            'weight_decay':weight_decay,
            'optimizer_type':optimizer,
            'scheduler_mode':scheduler,
            'T_max':comm_rounds,
            'save_model':True,
            'strategy':strategy,
            'ft_strategies':ft_strategies,
            'cifar':True,
            'data_part_seed':0,
            'transform_gen_seed':1,
            'dataset':dataset,
            'alpha':alpha,
            'iterations':iterations,
            'tr_config':tr_config,
            'partition':partition,
            'feature_dims':feature_dims,
            'label_smoothing':label_smoothing,
            'trn_logging':trn_logging,
            'ft_scheduler':ft_scheduler}
    
    return config

def get_finetuning_config(logdir, comment, degradation, model, model_type, dataset, cross_val_id=None, strategy=True):
    if dataset in ['cifar10', 'imagenet']:
        task = 'classification'
        batch_size = 64 if model == 'resnet18' else 64
        ft_comm_rounds = 400
    elif dataset in ['celeba', 'minicoco']:
        task = 'segmentation'
        batch_size = 32
        ft_comm_rounds = 50
    
    if degradation == 'classsep':
        if dataset == 'minicoco':
            partition = 'dirichlet'
            alpha = 0.1
        else:
            partition = 'by_class'
            alpha = None
    else:
        partition = 'dirichlet'
        alpha = 1e7

    tr_config = {'var_add':(0.005, 1),
                'alpha':(1.2, 1.5),
                'var_mul':(0.01, 0.5),
                'patch_size':3,
                'swap_count':1}
    
    if 'emb' in model_type:
        emb_dim = 128
        if model == 'resnet18':
            feature_dims = 62 * np.array([1, 2, 4, 8])
        elif model in ['convnext', 'convnextog', 'swinv2']:
            feature_dims = 90 * np.array([1, 2, 4, 8])
    else:
        emb_dim = None
        if model == 'resnet18':
            feature_dims = 64 * np.array([1, 2, 4, 8])
        elif model in ['convnext', 'convnextog', 'swinv2']:
            feature_dims = 96 * np.array([1, 2, 4, 8])
    fdim_str = str(64 if feature_dims is None else feature_dims[0])

    iterations = 50 if model == 'resnet18' else 50
    site_number = 5
    trn_site_number = 2
    lr = 1e-4
    ff_lr = 1e-4
    emb_lr = 0.01
    weight_decay = 1e-4 if model == 'resnet18' else 5e-2
    label_smoothing = 0. if model == 'resnet18' else 0.1

    optimizer = 'newadam' if model == 'resnet18'else 'adamw'
    ft_scheduler = 'cosine'

    trn_logging = True
    strategy = strategy if strategy is not None else 'embbnft'

    config = {'logdir':logdir,
            'comment':f'{comment}-{strategy}-{task}-{model}-{model_type}-{degradation}-s{str(site_number)}-ts{str(trn_site_number)}-edim{str(emb_dim)}-b{str(batch_size)}-ftcr{str(ft_comm_rounds)}-lr{str(lr)}-fflr{str(ff_lr)}-emblr{str(emb_lr)}-{optimizer}-{ft_scheduler}-wd{str(weight_decay)}-{dataset}-fdim{fdim_str}-ls{str(label_smoothing)}-xval{cross_val_id}',
            'task':task,
            'model_name':model,
            'model_type':model_type,
            'degradation':degradation,
            'site_number':site_number,
            'trn_site_number':trn_site_number,
            'embedding_dim':emb_dim,
            'batch_size':batch_size,
            'cross_val_id':cross_val_id,
            'ft_comm_rounds':ft_comm_rounds,
            'lr':lr,
            'ff_lr':ff_lr,
            'emb_lr':emb_lr,
            'weight_decay':weight_decay,
            'optimizer_type':optimizer,
            'ft_scheduler':ft_scheduler,
            'save_model':True,
            'strategy':strategy,
            'cifar':True,
            'data_part_seed':0,
            'transform_gen_seed':1,
            'dataset':dataset,
            'alpha':alpha,
            'iterations':iterations,
            'tr_config':tr_config,
            'partition':partition,
            'feature_dims':feature_dims,
            'label_smoothing':label_smoothing,
            'trn_logging':trn_logging,}

    return config