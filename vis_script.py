import os
import random

import numpy as np
import torch

from config import get_config, get_new_config
from main import main, new_main, new_main_plus_ft
from utils.val_and_vis import eval_points, get_points, load_embs

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_num_threads(8)

site = 40
fts = 8
deg = ['digits']
# comment = 'size2res20ftpca20trnembs'
comment = 'debug4'
logdir = '4digits'
comm_rounds = 1000
norm_layer = 'bn'
no_batch_running_stats = True
cl_per_site = 4

emb_dim = 64

strategy = 'pureemb'
model_type = 'embbn4'
fed_prox = 0
prox_map = False
ncc_lambda = 0.

main_fn = new_main_plus_ft

model_path = 'saved_models/4digits/2024_08_20-12_04_49-edim-pureemb-classification-resnet18-embbn4-di-s40-fts8-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4.state'
cross_val_id = None
gl_seed = 0

p_mode = 'grid'
saved_embs = load_embs('saved_models/4digits/2024_08_20-12_04_49-edim-pureemb-classification-resnet18-embbn4-di-s40-fts8-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4.state')
saved_ft_embs = load_embs('saved_models/4digits_ft/2024_08_21-00_36_28-edim-pureemb-classification-resnet18-embbn4-di-s40-fts8-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-64-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4-onlyemb.state')
vectors = np.zeros((site, emb_dim))
for i in range(site - fts):
    vectors[i] = saved_embs[i]
for i in range(site - fts, site):
    vectors[i] = saved_ft_embs[i - site + fts]

# p_mode = []
# p_mode.append(saved_ft_embs[0])
# p_mode.append(np.array([ -2.59954778,  -4.43291605,  -1.68389924,  -2.27556763,   5.7673016 ,   1.97867915,   2.69496957,   3.38415464,  -0.54656544,   5.74935743,
#         -5.40460361,  -0.10247165,   1.53854665,  -1.33875604,   2.49789027,  -4.41338756,   1.78467253,   2.66340665, -15.1502893 ,  -3.80348274,
#         -0.52785492,   8.57399005,  -0.62904113,   0.34870082,   2.45400634,  -2.0210656 ,   2.73353495,  -1.63989185,  -6.06217799,   1.56498503,
#          5.45550234,  -3.28350745,   5.9690504 ,   1.68628047,  -0.07324188,   1.9776351 ,   2.17626826,   0.83496682,  -0.58947227,   3.78839645,
#          2.66130751,   1.32051459,  -1.93679492,   7.06800322,  -4.13409978,  -1.17849983,   2.38605547,   7.20280421,  -3.86545741,   0.09726812,
#          0.4689508 ,   1.10979905,  -2.86089852,  -6.82635569,  -6.76025042,  -1.43148314,  -0.96197881,  -1.76197309,   4.37628171,  -1.33113644,
#          0.47744657,  10.88946206,  -1.65937522,   3.51509786]))
# p_mode.append(np.array([  3.88258488,  -3.47238247,  -1.49756677,  -2.16201441,  -7.23322052,   6.42577976,   1.85515268,  -6.89435725,   1.48936501,   5.20072579,
#          6.75573537,   2.07729122,  -8.86214788,   1.59606066,   0.51069937,  -0.91057831,  -8.40849065, -11.89361354,  -5.50079304,  -3.82269727,
#          6.65111506,   3.12642644,  12.10297434,   2.42034926,  -1.62695774,   0.84564205,  -8.86787162,  11.83358475,  10.83462982,  -3.44887707,
#          3.34243301,  -1.07694067,  -1.17365111, -10.99979471,   0.22621298,  -6.25852197,   0.31091249,  -7.01679603,  -3.7533633 ,  -1.06784777,
#         -1.30087951,   7.00200623,   4.82493879,  -6.48103462,   4.60335237,   4.16820735,  -2.71888092,  -3.20877723,  -1.50261453,  -5.95801384,
#         -1.69720687,  -5.62606348,  -3.87384445,   0.39296024,  -3.94946374,   1.30070382,   1.71266811, -12.22835628,  -2.85531278,  -1.5450177 ,
#          5.87796001, -10.39089412,   4.06575065,  -4.28983071]))
# vectors = None

torch.manual_seed(gl_seed)
random.seed(gl_seed)
np.random.seed(gl_seed)

def run():
    config = get_new_config(logdir, comment, site, deg, comm_rounds, strategy, model_type, fed_prox, prox_map, emb_dim=emb_dim, ft_site_number=fts, cross_val_id=cross_val_id, gl_seed=gl_seed, norm_layer=norm_layer, no_batch_running_stats=no_batch_running_stats, cl_per_site=cl_per_site, ncc_lambda=ncc_lambda)
    points, pca = get_points(model_path, points=p_mode, vectors=vectors, **config)
    losses, accuracies, embeddings, cl = eval_points(embeddings=points, model_path=model_path, device='cuda', **config)
    save_comm = config['comment']
    save_dir = os.path.join('loss_vis', logdir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'losses':losses,
                'accuracies':accuracies,
                'embeddings':embeddings,
                'classes':cl,
                'pca':pca},
                os.path.join(save_dir, save_comm))

if __name__ == '__main__':
    run()