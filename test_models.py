"""
Matching communication network with email network in the MC3 dataset
"""

import dev.util as util
from dev.util import logger
import matplotlib.pyplot as plt
from model.GromovWassersteinLearning import GromovWassersteinLearning
from model.BAPG import process_interaction_data
import numpy as np
import pickle
import torch.optim as optim
from torch.optim import lr_scheduler
import time


n_networks = 10
n_nodes = [25,50,100]
n_noises = 6

time_GWEMBED = {}
time_BAPG = {}

node_accuracy_GWEMBED = {}
node_accuracy_BAPG = {}

for n in n_nodes:
    for i in range(n_noises):
        time_GWEMBED[(n, i)] = []
        time_BAPG[(n, i)] = []
        node_accuracy_BAPG[(n, i)] = []
        node_accuracy_GWEMBED[(n, i)] = []


for nn in range(n_networks):
    for n in n_nodes:
        for i in range(n_noises):

            data_name = 'syn_{}_{}_{}'.format(nn, n, i)
            result_folder = 'match_syn'
            cost_type = ['cosine']
            method = ['proximal']

            util.makedirs(result_folder)

            filename = '{}/{}_database.pickle'.format(util.DATA_TRAIN_DIR, data_name)
            with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
                data_mc3 = pickle.load(f)

            print(len(data_mc3['src_index']))
            print(len(data_mc3['tar_index']))
            print(len(data_mc3['src_interactions']))
            print(len(data_mc3['tar_interactions']))

            connects = np.zeros((len(data_mc3['src_index']), len(data_mc3['src_index'])))
            for item in data_mc3['src_interactions']:
                connects[item[0], item[1]] += 1
            plt.imshow(connects)
            plt.savefig('{}/{}_src.png'.format(result_folder, data_name))
            plt.close('all')

            connects = np.zeros((len(data_mc3['tar_index']), len(data_mc3['tar_index'])))
            for item in data_mc3['tar_interactions']:
                connects[item[0], item[1]] += 1
            plt.imshow(connects)
            plt.savefig('{}/{}_tar.png'.format(result_folder, data_name))
            plt.close('all')

            opt_dict = {'epochs': 5,
                        'batch_size': 10000,
                        'use_cuda': False,
                        'strategy': 'soft',
                        'beta': 1e-1,
                        'outer_iteration': 400,
                        'inner_iteration': 1,
                        'sgd_iteration': 300,
                        'prior': False,
                        'prefix': result_folder,
                        'display': True}

            for m in method:
                for c in cost_type:
                    hyperpara_dict = {'src_number': len(data_mc3['src_index']),
                                      'tar_number': len(data_mc3['tar_index']),
                                      'dimension': 20,
                                      'loss_type': 'L2',
                                      'cost_type': c,
                                      'ot_method': m}

                    gwd_model = GromovWassersteinLearning(hyperpara_dict)

                    # initialize optimizer
                    optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-3)
                    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

                    print("\nRunning Gromov-Wasserstein learning {}".format(data_name))

                    # Gromov-Wasserstein learning
                    time_start = time.time()
                    gwd_model.train_without_prior(data_mc3, optimizer, opt_dict, scheduler=None)
                    time_end = time.time()
                    node_accuracy_GWEMBED[(n, i)].append(gwd_model.NC1)
                    time_GWEMBED[(n, i)].append(time_end - time_start)
                    print('Gromov-Wasserstein learning time cost: {:.4f}s'.format(time_end - time_start))

                    print("\nRunning BAPG {}".format(data_name))
                    # BAPG method
                    time_start = time.time()
                    results = process_interaction_data(data_mc3, track_accuracy=True, accuracy_interval=1)
                    time_end = time.time()
                    print('BAPG time cost: {:.4f}s'.format(time_end - time_start))

                    node_accuracy_BAPG[(n, i)].append(results['accuracy'])
                    time_BAPG[(n, i)].append(time_end - time_start)

                    matrix = results['coupling_matrix']
                    #only keep the max value in each row
                    for i in range(matrix.shape[0]):
                        row = matrix[i]
                        max_index = row.argmax()
                        new_row = [0] * matrix.shape[1]
                        new_row[max_index] = 1
                        matrix[i] = new_row

                    plt.imshow(matrix, interpolation='nearest')
                    plt.savefig('{}/trans_last_bapg_matching.png'.format(result_folder))
                    plt.close('all')


# Save results
import os

res_folder = os.path.join(os.getcwd(), "stats_data")
os.makedirs(res_folder, exist_ok=True)

result_filename = '{}/matching_results_syn.pickle'.format(res_folder)

with open(result_filename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump({'time_GWEMBED': time_GWEMBED,
                 'time_BAPG': time_BAPG,
                 'node_accuracy_GWEMBED': node_accuracy_GWEMBED,
                 'node_accuracy_BAPG': node_accuracy_BAPG}, f)
