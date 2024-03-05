def get_layer_list(task, strategy, original_list):
    if task == 'classification':
        if strategy == 'all':
            layer_list = original_list
        elif strategy == 'noembed':
            layer_list = [name for name in original_list if not ('embedding' in name.split('.') or 'fc' in name.split('.'))]
        elif strategy == 'nomerge':
            layer_list = []
        elif strategy == 'finetuning':
            layer_list = [name for name in original_list if 'embedding' in name.split('.') or 'fc' in name]
        elif strategy == 'fffinetuning':
            layer_list = [name for name in original_list if 'embedding' in name.split('.') or 'fc' in name or 'generator' in name]
        elif strategy == 'affinetoo':
            layer_list = [name for name in original_list if 'embedding' in name.split('.') or 'fc' in name or 'affine' in name.split('.')]
        elif strategy == 'onlyfc':
            layer_list = [name for name in original_list if 'fc' in name]
        elif strategy == 'onlyemb':
            layer_list = [name for name in original_list if 'embedding' in name.split('.')]
        elif strategy == 'l4-fc':
            layer_list = [name for name in original_list if 'fc' in name or 'layer4' in name.split('.')]
        elif strategy == 'extra_conv':
            layer_list = [name for name in original_list if 'conv0' in name or 'fc' in name]
        elif strategy == 'onlyextra_conv':
            layer_list = [name for name in original_list if 'conv0' in name]
    elif task == 'segmentation':
        if strategy == 'all':
            layer_list = original_list
        elif strategy == 'noembed':
            layer_list = [name for name in original_list if not ('embedding' in name or 'head.head' in name)]
        elif strategy == 'nomerge':
            layer_list = []
        elif strategy == 'onlyfc':
            layer_list = [name for name in original_list if 'head.head' in name]
        elif strategy == 'onlyemb':
            layer_list = [name for name in original_list if 'embedding' in name]
        elif strategy == 'finetuning':
            layer_list = [name for name in original_list if 'embedding' in name or 'head.head' in name]
        elif strategy == 'fffinetuning':
            layer_list = [name for name in original_list if 'embedding' in name or 'head.head' in name or 'generator' in name]

    return layer_list