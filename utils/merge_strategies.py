def get_layer_list(task, strategy, model):
    original_list = list(model.state_dict().keys())
    if task == 'classification':
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
        elif strategy == 'embandgen':
            layer_list = [name for name in original_list if not ('embedding' in name or 'generator' in name)]
        elif strategy == 'embgennorm':
            layer_list = [name for name in original_list if not ('embedding' in name or 'generator' in name or 'norm' in name)]
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
        elif strategy == 'fffinetuning':
            layer_list = [name for name in original_list if 'embedding' in name or 'fc' in name or 'generator' in name]
        elif strategy == 'affinetoo':
            layer_list = [name for name in original_list if 'embedding' in name or 'fc' in name or 'affine' in name]
        elif strategy == 'onlyfc':
            layer_list = [name for name in original_list if 'fc' in name]
        elif strategy == 'onlyemb':
            layer_list = [name for name in original_list if 'embedding' in name]
        elif strategy == 'l4-fc':
            layer_list = [name for name in original_list if 'fc' in name or 'layer4' in name]
        elif strategy == 'extra_conv':
            layer_list = [name for name in original_list if 'conv0' in name or 'fc' in name]
        elif strategy == 'onlyextra_conv':
            layer_list = [name for name in original_list if 'conv0' in name]
    elif task == 'segmentation':
        if strategy == 'all':
            layer_list = original_list
        elif strategy == 'noembed':
            layer_list = [name for name in original_list if not ('embedding' in name or 'head.head' in name)]
        elif strategy == 'fedbntrn':
            layer_list = [name for name in original_list if not ('embedding' in name or 'head.head' in name or 'norm' in name)]
        elif strategy == 'embbn':
            layer_list = [name for name in original_list if not ('embedding' in name or 'norm' in name)]
        elif strategy == 'nomerge':
            layer_list = []
        elif strategy == 'onlyfc':
            layer_list = [name for name in original_list if 'head.head' in name]
        elif strategy == 'onlyemb':
            layer_list = [name for name in original_list if 'embedding' in name]
        elif strategy == 'finetuning':
            layer_list = [name for name in original_list if 'embedding' in name or 'head.head' in name]
        elif strategy == 'fedbn':
            layer_list = [name for name in original_list if 'embedding' in name or 'head.head' in name or 'norm' in name]
        elif strategy == 'fffinetuning':
            layer_list = [name for name in original_list if 'embedding' in name or 'head.head' in name or 'generator' in name]

    return layer_list