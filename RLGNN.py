# %%
import csv
import os
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge

from utils import accuracy, sparse_mx_to_torch_sparse_tensor
from model import Encoder, Model
from time import perf_counter as t
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from collections import Counter

def get_low_idx(loss, labels, k=1.5):
    thresholds = []

    for category in labels.unique():
        category_losses = loss[labels == category]

        mean_loss = category_losses.mean()
        std_loss = category_losses.std()

        threshold = mean_loss + k * std_loss
        thresholds.append(threshold)

    noise_nodes = []
    clean_nodes = []

    for i, loss in enumerate(loss):
        category = labels[i]
        category_threshold = thresholds[category]
        if loss > category_threshold:
            noise_nodes.append(i)
        else:
            clean_nodes.append(i)


    return clean_nodes, noise_nodes

def get_low_and_gmm(z, pred_clean_idx, noise_labels, num_classes):
    data_np = z.cpu().detach().numpy()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_np)

    gmm = GaussianMixture(n_components=num_classes)

    gmm.fit(data_scaled)

    clusters = gmm.predict(data_scaled)

    cluster_labels = [[] for _ in range(num_classes)]

    for idx, cluster in enumerate(clusters):
        if idx in pred_clean_idx:
            cluster_labels[cluster].append(noise_labels[idx])

    final_labels = [Counter(labels).most_common(1)[0][0] if labels else -1 for labels in cluster_labels]
    matching_indices = [idx for idx, cluster in enumerate(clusters) if noise_labels[idx] == final_labels[cluster] and idx in pred_clean_idx]
    return matching_indices

def get_semi_loss(p, other_idx, t1=0.6, t2=0.3):
    loss = 0.0
    n_samples = 0

    for idx in other_idx:
        p_i = p[idx]

        top_probs, top_labels = torch.topk(p_i, 2)

        if top_probs[0] > t1:
            loss += F.cross_entropy(p_i.unsqueeze(0), top_labels[0].unsqueeze(0))
            n_samples += 1
        elif top_probs[1] > t2:
            total = top_probs[0] + top_probs[1]
            weight1 = top_probs[0] / total
            weight2 = top_probs[1] / total

            loss += weight1 * F.cross_entropy(p_i.unsqueeze(0), top_labels[0].unsqueeze(0))
            loss += weight2 * F.cross_entropy(p_i.unsqueeze(0), top_labels[1].unsqueeze(0))
            n_samples += 1

    return loss / n_samples if n_samples > 0 else torch.tensor(0.0)


class RLGNN:
    def __init__(self, config, device):
        self.weights = None
        self.edge_index = None
        self.device = device
        self.config = config
        self.best_val_acc = 0
        self.best_pred = None
        self.model = None
        self.pred_edge_index = None

    def fit(self, adj, features, idx_train, idx_val, idx_test, noise_labels, noisy_idx, clean_idx, clean_labels):
        config = self.config
        self.num_classes = noise_labels.max().item() + 1
        base_model = ({'GCNConv': GCNConv})[config['base_model']]
        activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
        encoder = Encoder(features.shape[1], config['num_hidden'], activation,
                             base_model, k=config['num_layers']).to(self.device)
        self.model = Model(encoder, config['num_hidden'], config['num_proj_hidden'],noise_labels.max() + 1,
                              config['tau']).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)
        self.edge_index = edge_index
        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        for epoch in range(1, config['num_epochs'] + 1):
            if epoch < 100:
                self.train1(epoch, self.model, features, edge_index)
            else:
                self.train2(epoch, self.model, features, edge_index, idx_train, idx_val,
                                  noise_labels, noisy_idx, clean_idx, clean_labels, idx_test)
        print("=== Final ===")
        self.model.load_state_dict(self.weights)

    def train1(self, epoch, model, x, edge_index):
        config = self.config
        model.train()
        self.optimizer.zero_grad()
        edge_index_1 = dropout_edge(edge_index, p=config['drop_edge_rate_1'])[0]
        edge_index_2 = dropout_edge(edge_index, p=config['drop_edge_rate_2'])[0]
        x_1 = self.drop_feature(x, config['drop_feature_rate_1'])
        x_2 = self.drop_feature(x, config['drop_feature_rate_2'])
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        h1 = model.projection(z1)
        h2 = model.projection(z2)
        loss = model.loss(h1, h2, batch_size=0).to(self.device)

        loss.backward()
        self.optimizer.step()
        print("epoch = {:d}, loss = {:05f}".format(epoch, loss.item()))

    def train2(self, epoch, model, x, edge_index, idx_train, idx_val, noise_labels, noisy_idx, clean_idx, clean_labels, idx_test):
        config = self.config
        model.train()
        labels = torch.from_numpy(noise_labels).long()
        self.optimizer.zero_grad()
        edge_index_1 = dropout_edge(edge_index, p=config['drop_edge_rate_1'])[0]
        edge_index_2 = dropout_edge(edge_index, p=config['drop_edge_rate_2'])[0]
        x_1 = self.drop_feature(x, config['drop_feature_rate_1'])
        x_2 = self.drop_feature(x, config['drop_feature_rate_2'])
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        h1 = model.projection(z1)
        h2 = model.projection(z2)
        loss1 = model.loss(h1, h2, batch_size=0).to(self.device)

        p1 = model.fc3(F.relu(h1))
        p2 = model.fc3(F.relu(h2))
        labels = labels.to(self.device)
        loss2 = F.cross_entropy(p1[idx_train], labels[idx_train]).to(self.device) + F.cross_entropy(p2[idx_train], labels[idx_train]).to(self.device)

        z = model(x, edge_index)
        h = model.projection(z)
        p = model.fc3(F.relu(h))
        ce = F.cross_entropy(p[idx_train], labels[idx_train], reduction='none').to(self.device)

        pred_clean_idx, pred_noise_idx = get_low_idx(ce, labels[idx_train], k=1)
        pred_clean_idx = get_low_and_gmm(z, pred_clean_idx, labels, self.num_classes)
        intersect_clean = set(pred_clean_idx) & set(clean_idx)
        intersect_noise = set(pred_noise_idx) & set(noisy_idx)
        intersect_clean = torch.tensor(list(intersect_clean))
        intersect_noise = torch.tensor(list(intersect_noise))
        
        TC = intersect_clean.shape[0] / len(pred_clean_idx)
        TN = intersect_noise.shape[0] / len(pred_noise_idx)
        print('TC = {:05f}'.format(TC))
        print('TN = {:05f}'.format(TN))
        print(intersect_clean.shape[0], len(pred_clean_idx))
        print(intersect_noise.shape[0], len(pred_noise_idx))
        loss3 = F.cross_entropy(p[pred_clean_idx], labels[pred_clean_idx]).to(self.device)
        other_idx = list(set(range(p.shape[0])) - set(pred_clean_idx))
        loss4 = get_semi_loss(p, other_idx, t1=config['t1'], t2=config['t2']).to(self.device)

        loss = loss1 + config['alpha'] * loss2 + loss3 + config['beta'] * loss4
        loss.backward()
        model.eval()
        val_acc = accuracy(p[idx_val], labels[idx_val])
        test_acc = accuracy(p[idx_test], labels[idx_test])
        print("epoch ={:d}, val acc ={:05f}, test acc ={:05f}".format(epoch, val_acc, test_acc))
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.weights = deepcopy(model.state_dict())

        self.optimizer.step()

    def test(self, args, model, x, y, idx_train, idx_val, idx_test):
        model.eval()
        if sp.issparse(x):
            x = sparse_mx_to_torch_sparse_tensor(x).to_dense().float()
        else:
            x = torch.FloatTensor(np.array(x))
        x = x.to(self.device)
        z = model(x, self.edge_index)
        h = model.projection(z)
        p = model.fc3(F.relu(h))
        acc = accuracy(p[idx_test], y[idx_test])
        print("test acc = ", acc.item())
        result = acc.item()
        path = "./output/RLGNN.csv"
        if not os.path.exists(path):
            with open(path,'w') as f:
                csv_write = csv.writer(f)
                csv_head = ['dataset', 'noise_type', 'noise_rate', 'acc']
                csv_write.writerow(csv_head)
        with open(path,'a+') as f:
            csv_write = csv.writer(f)
            data_row = [args.dataset, args.noise, args.ptb_rate, result]
            csv_write.writerow(data_row)


    def drop_feature(self, x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0
        return x