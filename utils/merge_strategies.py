def get_layer_list(strategy, model):
    original_list = list(model.state_dict().keys())
    if strategy == 'all':
        layer_list = original_list
    elif strategy == 'noembed':
        layer_list = [name for name in original_list if not ('embedding' in name or 'fc' in name)]
    elif strategy == 'pureemb':
        layer_list = [name for name in original_list if not ('embedding' in name)]
    elif strategy == 'fedbntrn':
        layer_list = [name for name in original_list if not ('embedding' in name or 'fc' in name or 'norm' in name)]
    elif strategy == 'embbn':
        layer_list = [name for name in original_list if not ('embedding' in name or 'norm' in name)]
    elif strategy == 'nomerge':
        layer_list = []
    elif strategy == 'embandgenft':
        layer_list = [name for name in original_list if 'embedding' in name or 'generator' in name]
    elif strategy == 'embgennormft':
        layer_list = [name for name in original_list if 'embedding' in name or 'generator' in name or 'norm' in name]
    elif strategy == 'finetuning':
        layer_list = [name for name in original_list if 'embedding' in name or 'fc' in name]
    elif strategy == 'fedbn':
        layer_list = [name for name in original_list if 'embedding' in name or 'fc' in name or 'norm' in name]
    elif strategy == 'embbnft':
        layer_list = [name for name in original_list if 'embedding' in name or 'norm' in name]
    elif strategy == 'onlyemb':
        layer_list = [name for name in original_list if 'embedding' in name]

    return layer_list