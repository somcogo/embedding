import math
import time
import os

import torch
from torch import nn
from torch import optim


class EmbModel(nn.Module):
    def __init__(self, trained_features, model_config):
        super().__init__()

        forw_maps = []
        for key, feature in trained_features.items():
            feature_type = key.split('.')[-1]
            assert feature_type in['weight', 'bias']
            init_value = 1 if feature_type == 'weight' else 0
            forw_maps.append(ForwardLayer(feature.shape[1], init_value, model_config))
        self.forw_maps = nn.ModuleList(forw_maps)

        self.embs = nn.Parameter(torch.eye(trained_features[key].shape[0], model_config['emb_dim']))

    def forward(self, _):
        out = []
        for forw_map in self.forw_maps:
            out.append(forw_map(self.embs))
        return out

class ForwardLayer(nn.Module):
    def __init__(self, feature_dim, init_value, model_config):
        super().__init__()
        gen_depth = model_config['depth']

        layers = []
        for depth in range(gen_depth):
            in_dim = model_config['emb_dim'] if depth == 0 else model_config['feat_dim'] * 2**(depth - 1)
            out_dim = feature_dim if depth == gen_depth - 1 else model_config['feat_dim'] * 2**(depth)
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            if depth < gen_depth - 1:
                layers.append(nn.ReLU())

        fan_in = layers[-1].in_features
        bound_l = init_value - 1 / math.sqrt(fan_in)
        bound_r = init_value + 1 / math.sqrt(fan_in)
        nn.init.uniform_(layers[-1].bias, a=bound_l, b=bound_r)
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_trained_features(model_paths):
    site_state_dicts = []
    for path in model_paths:
        trn_file = torch.load(path, map_location='cpu')
        site_state_dicts.extend([trn_file[i]['site_model_state'] for i in range(len(trn_file.keys()) - 2)])

    trained_features = {}
    for key in list(site_state_dicts[0].keys()):
        feature = torch.stack([d[key] for d in site_state_dicts])
        trained_features[key] = feature

    return trained_features



def train(emb_model, opt, trained_features, trn_config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb_model.to(device)
    for k, f in trained_features.items():
        trained_features[k] = f.to(device)
    log = {'loss':[],
           'emb':[]}
    loss_fn = nn.MSELoss(reduction='none')
    min_loss = 1e5
    it = 0
    min_it = 0
    while it - min_it < 200:
        opt.zero_grad()
        pred = emb_model(torch.empty(1, device=device))
        losses = torch.zeros((len(trained_features), emb_model.embs.shape[0]), device=device)
        for i, feat in enumerate(trained_features.values()):
            losses[i] = loss_fn(pred[i], feat).mean(dim=1)
        sum_loss = losses.sum()
        sum_loss.backward()
        opt.step()
        if min_loss - 1e-2 > sum_loss.cpu():
            min_loss = sum_loss.detach().cpu()
            min_it = it

        log['loss'].append(losses.detach().cpu())
        log['emb'].append(emb_model.embs.detach().cpu())
        it += 1
    
    log['loss'] = torch.stack(log['loss'])
    log['emb']  = torch.stack(log['emb'])
    log['pred'] = [p.detach().cpu() for p in pred]
    log['keys'] = list(trained_features.keys())
    return log


def main(model_path, trn_config, model_config, log_dir, comment):
    trained_features = get_trained_features(model_path)
    emb_model = EmbModel(trained_features=trained_features, model_config=model_config)
    emb_param = [p for n, p in emb_model.named_parameters() if 'embs' in n]
    other_param = [p for n, p in emb_model.named_parameters() if 'embs' not in n]
    opt = optim.Adam([{'params':emb_param, 'lr':trn_config['emb_lr']},
                      {'params':other_param}],
                      lr=trn_config['lr'])
    log = train(emb_model, opt, trained_features, trn_config)
    os.makedirs('emb_finetune', exist_ok=True)
    os.makedirs(os.path.join('emb_finetune', log_dir), exist_ok=True)
    torch.save(log, os.path.join('emb_finetune', log_dir, comment))



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model_paths = ['saved_models/alpha/2024_09_03-16_37_04-compare-embbn-classification-resnet18-vanilla-adcoal-s30-fts6-b64-commr1000-iter50-lr1e-4-fflrNone-elrNone-ftelrNone-embdim-None-cifar10-fedp-0.0-proxm-False-xvalNone-gls0-nl-in-rst-True-clprNone-ncc0.0.state',
                    'saved_models/alpha_ft/2024_09_04-00_06_09-compare-embbn-classification-resnet18-vanilla-adcoal-s30-fts6-b64-commr1000-iter50-lr1e-4-fflrNone-elrNone-ftelrNone-embdim-None-cifar10-fedp-0.0-proxm-False-xvalNone-gls0-nl-in-rst-True-clprNone-ncc0.0-embbnft.state']
    # edim = 512
    lr = 1e-4
    elr = 1e-2
    depth = 3
    log_dir = 'difflr'
    for edim in [2, 8, 32, 128, 512, 2048]:
        t1 = time.time()
        trn_config = {'iter':None,
                    'lr':lr,
                    'emb_lr':elr}
        model_config = {'emb_dim':edim,
                        'depth':depth,
                        'feat_dim':64}
        comment = f'lr{lr}elr{elr}edim{edim}depth{depth}'
        main(model_paths, trn_config, model_config, log_dir, comment)
        t2 = time.time()
        print('edim', edim, 'depth', depth, 'lr', lr, 'emb lr', elr, f'{t2-t1:0.3f}')