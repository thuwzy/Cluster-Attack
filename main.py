import torch
import torch.nn as nn

import argparse
import copy
import numpy as np
import time
import random

from target_model import *
from multi_feature_attack import *

parser = argparse.ArgumentParser()
parser.add_argument("--num_fake_nodes", type=int, default=4,  help="number of injected fake nodes")
parser.add_argument("--num_victim_nodes", type=int, default=10,  help="number of victim nodes")
parser.add_argument("--attack_round", type=int, default=100,  help="the number of repeat experiments")
parser.add_argument("--alpha", type=float, default=1.0,  help="query ratio")
parser.add_argument("--lambdaN", type=float, default=0.0,  help="trade-off parameter between loss function of victim nodes and protected nodes.")
parser.add_argument("--dataset", type=str, default='cora',  help="cora or citeseer")
args = parser.parse_args()

class MyFeature(nn.Module):
    def __init__(self, weight):
        super(MyFeature, self).__init__()
        self.weight = weight
    
    def forward(self, nodes):
        return self.weight[nodes]

if __name__ == "__main__":
    np.random.seed(5253)
    random.seed(2345)
    torch.manual_seed(6542)
    torch.cuda.manual_seed_all(1964)

    lambdaN=args.lambdaN
    num_fake_nodes = args.num_fake_nodes
    edge_limit = args.num_victim_nodes
    num_attack = args.num_victim_nodes

    if args.dataset == 'cora':
        feat_data, labels, adj_lists, num_nodes, num_feats, num_label = load_cora(num_fake_nodes)
    else:
        feat_data, labels, adj_lists, num_nodes, num_feats, num_label = load_citeseer(num_fake_nodes)
    origin_adj_lists = copy.deepcopy(adj_lists)
    
    features = MyFeature(torch.FloatTensor(feat_data))

    if args.dataset == 'cora':
        target_model = run_cora(features, labels, origin_adj_lists, num_nodes, num_fake_nodes, num_feats)
    else:
        target_model = run_citeseer(features, labels, origin_adj_lists, num_nodes, num_fake_nodes, num_feats)

    target_model.eval()
    for p in target_model.parameters():
        p.require_grads = False

    cnt_all = 0.0
    cnt_suc = 0.0
    cnt_all_nei = 0.0
    cnt_suc_nei = 0.0
    start_time = time.time()
    acc_for_var = []
    Kt_sample = int(args.alpha*features.weight.shape[1])
    Kf_sample = int(args.alpha*features.weight.shape[1])

    for _ in range(args.attack_round):
        if (_+1)%5==0: # retrain the model for every 5 runs
            if args.dataset == 'cora':
                target_model = run_cora(features, labels, origin_adj_lists, num_nodes, num_fake_nodes, num_feats)
            else:
                target_model = run_citeseer(features, labels, origin_adj_lists, num_nodes, num_fake_nodes, num_feats)
        
        acc, real_num_attack, nei_acc, nei_num, D_attack_nodes, S_attacked_nodes = cluster_attack(target_model, features, labels, origin_adj_lists, adj_lists, num_nodes, num_feats, num_label, num_fake_nodes, num_attack, edge_limit, Kt_sample, Kf_sample, lambdaN)

        acc_for_var.append(acc)

        cnt_suc += float(acc) * float(real_num_attack)
        cnt_all += float(real_num_attack)
        cnt_suc_nei += float(nei_acc) * float(nei_num)
        cnt_all_nei += float(nei_num)
        print("Round[%d], Attacked %d nodes, success %d nodes, succ rate = %f, acc of neigh nodes = %f, std = %f, num_of_attacked_nodes = %f, total time = " %(_, cnt_all, cnt_suc, float(cnt_suc)/float(cnt_all), float(cnt_suc_nei)/float(cnt_all_nei), np.std(acc_for_var), float(cnt_all)/float(_+1)), time.time() - start_time)


    

