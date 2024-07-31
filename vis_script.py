import os
import random

import numpy as np
import torch

from config import get_config, get_new_config
from main import main, new_main, new_main_plus_ft
from utils.val_and_vis import eval_points, get_points, load_embs

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_num_threads(8)

site = 20
fts = 4
deg = 'classshard'
comment = 'size1-5res10trnftpca20-4site'
# comment = 'ftembeval'
logdir = '4clpersite'
comm_rounds = 1000
norm_layer = 'bn'
no_batch_running_stats = True
cl_per_site = 4

emb_dim = 4

strategy = 'pureemb'
model_type = 'embbn4'
fed_prox = 0
prox_map = False

main_fn = new_main_plus_ft

model_path = 'saved_models/seededruns/2024_07_18-17_23_17-moresites-pureemb-classification-resnet18-embbn4-classshard-s20-fts4-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-4-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4.state'
cross_val_id = None
gl_seed = 0

p_mode = 'grid'
saved_ft_embs = load_embs('saved_models/seededruns_ft/2024_07_18-23_11_23-moresites-pureemb-classification-resnet18-embbn4-classshard-s20-fts4-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-4-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4-onlyemb.state')
saved_embs = load_embs('saved_models/seededruns/2024_07_18-17_23_17-moresites-pureemb-classification-resnet18-embbn4-classshard-s20-fts4-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-4-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4.state')
vectors = np.zeros((10, 4))
for i in range(8):
    vectors[i] = saved_embs[i]
vectors[8] = saved_ft_embs[0]
vectors[9] = saved_ft_embs[1]

torch.manual_seed(gl_seed)
random.seed(gl_seed)
np.random.seed(gl_seed)

def run():
    config = get_new_config(logdir, comment, site, deg, comm_rounds, strategy, model_type, fed_prox, prox_map, emb_dim=emb_dim, ft_site_number=fts, cross_val_id=cross_val_id, gl_seed=gl_seed, norm_layer=norm_layer, no_batch_running_stats=no_batch_running_stats, cl_per_site=cl_per_site)
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

    # import os
    # import math
    # import logging

    # import torch
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib import cm

    # from config import get_new_config
    # from utils.val_and_vis import load_embs, pca_grid, load_embs
    # logging.getLogger('matplotlib.pyplot').setLevel(logging.INFO)
    # saved_embs = load_embs('saved_models/seededruns/2024_07_18-17_23_43-moresites-pureemb-classification-resnet18-embbn4-classshard-s10-fts2-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-4-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4.state')
    # saved_ft_embs = load_embs('saved_models/seededruns_ft/2024_07_18-20_11_13-moresites-pureemb-classification-resnet18-embbn4-classshard-s10-fts2-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-4-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4-onlyemb.state')
    # loss_dict = torch.load('loss_vis/4clpersite/size1-5res10trnftpca-pureemb-classification-resnet18-embbn4-classshard-s10-fts2-b64-commr1000-iter50-lr1e-4-fflr0.0001-elr0.1-ftelr0.1-embdim-4-cifar10-fedp-0-proxm-False-xvalNone-gls0-nl-bn-rst-True-clpr4')
    # pca = loss_dict['pca']
    # saved_embs_pca = pca.transform(saved_embs)
    # ft_embs_pca = pca.transform(saved_ft_embs)
    # embs = loss_dict['embeddings']
    # embs_np = np.array([np.array(e) for e in embs])
    # embs_pca = pca.transform(embs_np)
    # cl_number = len(loss_dict['classes'][0])
    # fig, ax = plt.subplots(cl_number + 1, len(loss_dict['classes']))
    # fig.set_size_inches(len(loss_dict['classes'])*2, 2*cl_number)
    # px_count = int(math.sqrt(embs_np.shape[0]))
    # for site in range(2):
    #     img = np.zeros((px_count, px_count))
    #     x = embs_pca[::px_count,0].squeeze()
    #     y = embs_pca[:px_count,1].squeeze()
    #     pic_cm = None
    #     dot_cm = cm.copper
    #     for i in range(px_count):
    #         for j in range(px_count):
    #             img[i, j] = math.log(loss_dict['losses'][px_count*i+j][site])
    #     ax[0, site].pcolormesh(x, y, img, shading='nearest', cmap=pic_cm)
    #     ax[0, site].scatter(ft_embs_pca[0,0], ft_embs_pca[0,1], c='yellow')
    #     ax[0, site].scatter(ft_embs_pca[1,0], ft_embs_pca[1,1], c='red')
    #     ax[0, site].scatter(saved_embs_pca[:,0], saved_embs_pca[:,1], c=range(2, 10), cmap=dot_cm)
    #     ax[0, site].set_title(f'Log loss, site {site}')

    #     # img = np.zeros((px_count, px_count))
    #     # acc = loss_dict['accuracies'][:, site, 0]
    #     # for i in range(px_count):
    #     #     for j in range(px_count):
    #     #         img[i, j] = acc[px_count*i+j]
    #     # ax[1, site].pcolormesh(x, y, img, shading='nearest', vmin=0, vmax=1, cmap=pic_cm)
    #     # ax[1, site].scatter(ft_embs_pca[0,0], ft_embs_pca[0,1], c='yellow')
    #     # ax[1, site].scatter(ft_embs_pca[1,0], ft_embs_pca[1,1], c='red')
    #     # ax[1, site].scatter(saved_embs_pca[:,0], saved_embs_pca[:,1], c=range(2, 10), cmap=dot_cm)
    #     # cl = loss_dict['classes'][site][0]
    #     # ax[1, site].set_title(f'Acc, cl {cl}, max {acc.max()}')

    #     for cl in range(cl_number):
    #         img = np.zeros((px_count, px_count))
    #         acc = loss_dict['accuracies'][:, site, cl]
    #         for i in range(px_count):
    #             for j in range(px_count):
    #                 img[i, j] = acc[px_count*i+j]
    #         ax[cl+1, site].pcolormesh(x, y, img, shading='nearest', vmin=0, vmax=1, cmap=pic_cm)
    #         ax[cl+1, site].scatter(ft_embs_pca[0,0], ft_embs_pca[0,1], c='yellow')
    #         ax[cl+1, site].scatter(ft_embs_pca[1,0], ft_embs_pca[1,1], c='red')
    #         ax[cl+1, site].scatter(saved_embs_pca[:,0], saved_embs_pca[:,1], c=range(2, 10), cmap=dot_cm)
    #         cla = loss_dict['classes'][site][cl]
    #         ax[cl+1, site].set_title(f'Acc, cl {cla}, max {acc.max()}')
    # fig.tight_layout()
    # plt.savefig('loss_vis/4clpersite.png')