import argparse
import random

import numpy as np
import yaml
from yaml import SafeLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from dataset import Dataset
import torch_geometric.utils as utils
from RLGNN import RLGNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--ptb_rate', type=float, default=0.2,
                        help="noise ptb_rate")
    parser.add_argument('--label_rate', type=float, default=0.05,
                        help="noise label_rate")
    parser.add_argument('--noise', type=str, default='uniform',
                        choices=['uniform', 'pair'],
                        help='type of noises')
    args = parser.parse_known_args()[0]

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    if args.dataset == 'photo':
        from torch_geometric.datasets import Amazon
        import torch_geometric.utils as utils
        dataset = Amazon('./data', args.dataset)
        adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
        features = dataset.data.x.numpy()
        labels = dataset.data.y.numpy()
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        idx_test = idx[:int(0.8 * len(labels))]
        idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
        idx_train = idx[int(0.9 * len(labels)):int((0.9 + args.label_rate) * len(labels))]
    else:
        data = Dataset(root='./data', name=args.dataset)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_train = idx_train[:int(args.label_rate * adj.shape[0])]


    from utils import noisify_with_P

    ptb = args.ptb_rate
    nclass = labels.max() + 1
    train_labels = labels[idx_train]
    noise_y, P, noisy_idx, clean_idx = noisify_with_P(y_train = train_labels, 
                                                    train_num = len(idx_train),
                                                    nb_classes = nclass, 
                                                    noise = ptb, 
                                                    random_state=10,  
                                                    noise_type=args.noise)
    noise_labels = labels.copy()
    noise_labels[idx_train] = noise_y
    use_cuda=True
    use_cuda = (torch.cuda.is_available() and use_cuda)
    device='cpu'
    device_count = torch.cuda.device_count()
    print("可用的 GPU 数量:", device_count)
    if use_cuda:
        if device_count >= 2:
            device = torch.device(1)  
        else:
            device = torch.device(0)  
    print ("use CUDA:", use_cuda, "- device:", device)
    print(device)
    sgnn = RLGNN(config, device)
    sgnn.fit(adj, features, idx_train, idx_val, idx_test, noise_labels, noisy_idx, clean_idx, labels)
    sgnn.test(args, sgnn.model, features, labels, idx_train, idx_val, idx_test)