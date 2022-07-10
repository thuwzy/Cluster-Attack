import torch
from torch.autograd import Variable
import numpy as np
import time
import random
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

'''
This file includes the code for Cluster Attack algorithm and related util functions.
'''

is_targeted = True

def reset_adj(fake_nodes, adj_lists):
    """
    Reset fake nodes' connections.
    """
    for node1 in fake_nodes: # connections between fake nodes
        for node2 in fake_nodes:
            if node1< node2:
                if node2 in adj_lists[node1]:
                    assert node1 in adj_lists[node2]
                    adj_lists[node1].remove(node2)
                    adj_lists[node2].remove(node1)   

    for fake_node_id in fake_nodes:
        tmp = adj_lists[fake_node_id].copy()
        for neigh_id in tmp:
            adj_lists[neigh_id].remove(fake_node_id)
            adj_lists[fake_node_id].remove(neigh_id)

def init_AV_adj(fake_node_id, attack_node_id, fake_nodes, attack_nodes, adj_lists, features, origin_adj_lists):
    """
    Connect the fake node with a victim node, and let other victim nodes isolated.
    """
    # reset the connections of fake nodes
    reset_adj(fake_nodes, adj_lists)

    # connect fake node with the victim node
    adj_lists[fake_node_id].add(attack_node_id)
    adj_lists[attack_node_id].add(fake_node_id)

def init_AV_cluster(fake_nodes, attack_nodes, adj_lists, features, node_to_cluster_id, origin_adj_lists):
    '''
    Connect victim nodes with fake nodes according to their clusters.
    '''
    # reset the connections of fake nodes
    reset_adj(fake_nodes, adj_lists)

    for attack_node_id in attack_nodes:
        if node_to_cluster_id[attack_node_id] == -1:
            continue
        adj_lists[node_to_cluster_id[attack_node_id]].add(attack_node_id) # connect fake node with victim node
        adj_lists[attack_node_id].add(node_to_cluster_id[attack_node_id])

def init_fake_feature(num_feats, fake_nodes, attack_nodes, target_label, adj_lists, features, node_to_AV, node_to_cluster_id):
    '''
    Initialize fake nodes' feature as the cluster center
    '''
    for fake_node_id in fake_nodes:
        cnt = 0.0
        features.weight[fake_node_id,:] = torch.FloatTensor(list(map(float, np.random.randint(1, size=num_feats)))) # initialize as all-0
        for attack_node_id in attack_nodes:
            if node_to_cluster_id[attack_node_id] == fake_node_id:
                cnt += 1.0
                features.weight[fake_node_id,:] += node_to_AV[attack_node_id]
        if cnt == 0:
            continue
        features.weight[fake_node_id,:] /= cnt
        features.weight[fake_node_id,:] = torch.round(features.weight[fake_node_id,:])

def multi_optimize_feature_attack(target_model, num_feats, fake_nodes, attack_nodes, target_label, predict_label, adj_lists, features, node_to_AV, node_to_cluster_id, num_label, Kf_sample, neigh_nodes, lambdaN):

    max_scores = target_model.forward(attack_nodes, adj_lists, features)
    max_scoresN = target_model.forward(neigh_nodes, adj_lists, features)
    if is_targeted:
        min_loss = CWLoss(max_scores, t_label=Variable(torch.LongTensor(target_label[attack_nodes])).squeeze(dim=1), num_label=num_label) + lambdaN*CWLoss(max_scoresN, t_label=Variable(torch.LongTensor(predict_label[neigh_nodes])).squeeze(dim=1), num_label=num_label)
    else:
        raise NotImplementedError()

    for fake_node_id in fake_nodes:
        features_to_try = list(range(num_feats))
        random.shuffle(features_to_try)
        features_to_try=features_to_try[:Kf_sample] # Kf_sample = the number of queries for each fake node

        for fea_idx in features_to_try:
            features.weight[fake_node_id, fea_idx] = 1 - features.weight[fake_node_id, fea_idx]
            scores = target_model.forward(attack_nodes, adj_lists, features)
            scoresN = target_model.forward(neigh_nodes, adj_lists, features)
            if is_targeted:
                loss = CWLoss(scores, t_label=Variable(torch.LongTensor(target_label[attack_nodes])).squeeze(dim=1), num_label=num_label) + lambdaN*CWLoss(scoresN, t_label=Variable(torch.LongTensor(predict_label[neigh_nodes])).squeeze(dim=1), num_label=num_label)
            else:
                raise NotImplementedError()

            if loss < min_loss:
                max_scores = scores.clone().detach()
                max_scoresN = scoresN.clone().detach()
                min_loss = loss.clone().detach()
            else:
                features.weight[fake_node_id, fea_idx] = 1 - features.weight[fake_node_id, fea_idx]

    return max_scores, max_scoresN, min_loss

def compute_AV(target_model, num_feats, fake_node_id, fake_nodes, attack_nodes, target_label, predict_label, adj_lists, features, num_label, Kt_sample, neigh_nodes, lambdaN):
    '''
    Compute Adversarial Vulnerability for a single victim node
    '''

    features.weight[fake_node_id,:] = torch.FloatTensor(list(map(float, np.random.randint(2, size=num_feats)))) # Random initialize
    max_scores = target_model.forward(attack_nodes, adj_lists, features) # victim nodes
    max_scoresN = target_model.forward(neigh_nodes, adj_lists, features) # protected nodes
    if is_targeted:
        min_loss = CWLoss(max_scores, t_label=Variable(torch.LongTensor(target_label[attack_nodes])).squeeze(dim=1), num_label=num_label) + lambdaN*CWLoss(max_scoresN, t_label=Variable(torch.LongTensor(predict_label[neigh_nodes])).squeeze(dim=1), num_label=num_label)
    else:
        raise NotImplementedError()

    features_to_try = list(range(num_feats))
    random.shuffle(features_to_try)
    features_to_try=features_to_try[:Kt_sample]

    for fea_idx in features_to_try:
        features.weight[fake_node_id, fea_idx] = 1 - features.weight[fake_node_id, fea_idx]
        
        scores = target_model.forward(attack_nodes, adj_lists, features)
        scoresN = target_model.forward(neigh_nodes, adj_lists, features)
        if is_targeted:
            loss = CWLoss(scores, t_label=Variable(torch.LongTensor(target_label[attack_nodes])).squeeze(dim=1), num_label=num_label) + lambdaN*CWLoss(scoresN, t_label=Variable(torch.LongTensor(predict_label[neigh_nodes])).squeeze(dim=1), num_label=num_label)
        else:
            raise NotImplementedError()

        if loss < min_loss:
            max_scores = scores.clone().detach()
            max_scoresN = scoresN.clone().detach()
            min_loss = loss.clone().detach()
        else:
            features.weight[fake_node_id, fea_idx] = 1 - features.weight[fake_node_id, fea_idx]

    tmp = features.weight[fake_node_id,:].clone().detach()
    return tmp

def compute_AV_dict(target_model, num_feats, fake_nodes, attack_nodes, target_label, predict_label, adj_lists, origin_adj_lists, features, num_label, Kt_sample, neigh_nodes, lambdaN):
    fake_node_id = fake_nodes[0] # use the 0th fake node
    node_to_AV = {}
    for attack_node_id in attack_nodes:
        init_AV_adj(fake_node_id, attack_node_id, fake_nodes, attack_nodes, adj_lists, features, origin_adj_lists)
        tmp1 = compute_AV(target_model, num_feats, fake_node_id, fake_nodes, attack_nodes, target_label, predict_label, adj_lists, features, num_label, Kt_sample, neigh_nodes, lambdaN)
        node_to_AV[attack_node_id] = tmp1
    return node_to_AV

def compute_AV_cluster(fake_nodes, attack_nodes, target_label, adj_lists, features, node_to_AV, edge_limit, Kt_sample):

    node_to_cluster_id = {} # describe each victim belongs to which cluster, input: victim node id, output: fake node id

    estimator = KMeans(n_clusters=len(fake_nodes))
    AVs = np.zeros((len(attack_nodes), len(features.weight[0])))
    for i, attack_node in enumerate(attack_nodes):
        AVs[i] = node_to_AV[attack_node].cpu()
    estimator.fit(AVs)

    label_pred = estimator.labels_
    for i, attack_node in enumerate(attack_nodes):
        node_to_cluster_id[attack_node] = fake_nodes[label_pred[i]]
    return node_to_cluster_id

def CWLoss(score, t_label, num_label):
    '''
    score = [
        [0.1, 0.4, ..., -0.1],
        ...,
        [0.1, 0.4, ..., -0.1]
    ]
    t_label = [1, 3, 0, ..., 1] # target label
    '''
    one_hot = torch.zeros(t_label.shape[0], num_label).scatter_(1, t_label.unsqueeze(1), 1) * (10**10) # inf = 10**10
    score_copy = score.clone().detach()
    score_copy = score_copy - one_hot # this is used to compute the second largest

    if is_targeted:
        mask = (score_copy.max(1)[0] - torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1)).ge(0).float() # whether attack successed
        # sqrt trick is adopted here
        cw_loss = torch.sqrt(((score_copy.max(1)[0] - torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1)) * mask)).sum(0)
    else:
        mask = (torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1) - score_copy.max(1)[0]).ge(0).float()
        cw_loss = ((torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1) - score_copy.max(1)[0]) * mask).sum(0)
    return cw_loss

def cluster_attack(target_model, features, labels, origin_adj_lists, adj_lists, num_nodes, num_feats, num_label, num_fake_nodes, num_attack, edge_limit, Kt_sample, Kf_sample, lambdaN):
    '''
    Cluster Attack
    '''
    fake_nodes = np.array(range(num_nodes, num_nodes+num_fake_nodes)) # list of fake nodes
    reset_adj(fake_nodes, adj_lists) # reset fake nodes' connection

    nodes_list = np.array(range(num_nodes)) # [0, 1, ..., numnodes-1]
    predict_label = torch.from_numpy(target_model.forward(nodes_list, adj_lists, features).data.detach().cpu().numpy().argmax(axis=1)).unsqueeze(dim=1).numpy() # predicted label
    correct_nodes = list(nodes_list[:]) # All nodes are correct now.

    random.shuffle(correct_nodes) # Shuffle the nodes.
    attack_nodes = correct_nodes[:num_attack] # list of victim nodes

    neigh_nodes = [] # protected nodes
    attack_nodes_copy = attack_nodes.copy()
    for attack_node_id in attack_nodes_copy:
        for neigh_id in adj_lists[attack_node_id]:
            if (neigh_id not in attack_nodes) and (neigh_id not in fake_nodes) and (neigh_id not in neigh_nodes):
                neigh_nodes.append(neigh_id)

    D_attacked_nodes = [] # degrees of attacked nodes
    for attacked_node in attack_nodes:
        D_attacked_nodes.append(len(adj_lists[attacked_node]))

    target_label = (labels[:] + 2) % num_label # We perform targeted attack. Here we set the target label.

    node_to_AV = compute_AV_dict(target_model, num_feats, fake_nodes, attack_nodes, target_label, predict_label, adj_lists, origin_adj_lists, features, num_label, Kt_sample, neigh_nodes, lambdaN) # 储存每个节点的AV

    node_to_cluster_id = compute_AV_cluster(fake_nodes, attack_nodes, target_label, adj_lists, features, node_to_AV, edge_limit, Kt_sample)
    
    init_AV_cluster(fake_nodes, attack_nodes, adj_lists, features, node_to_cluster_id, origin_adj_lists) # Connect fake nodes with victim nodes

    init_fake_feature(num_feats, fake_nodes, attack_nodes, target_label, adj_lists, features, node_to_AV, node_to_cluster_id)

    start_time = time.time()
    scores, scoresN, loss = multi_optimize_feature_attack(target_model, num_feats, fake_nodes, attack_nodes, target_label, predict_label, adj_lists, features, node_to_AV, node_to_cluster_id, num_label, Kf_sample, neigh_nodes, lambdaN)
    end_time = time.time()

    if is_targeted:
        succ = accuracy_score(target_label[attack_nodes], scores.data.detach().cpu().numpy().argmax(axis=1))
        nei_acc = accuracy_score(predict_label[neigh_nodes], scoresN.data.detach().cpu().numpy().argmax(axis=1))
    else:
        raise NotImplementedError()

    print("Attack Success Rate:", succ, "time =", end_time-start_time)

    S_attacked_nodes = [] # successness of attacked nodes
    predicted_labels = scores.data.detach().cpu().numpy().argmax(axis=1)
    target_labels = target_label[attack_nodes]
    for idx in range(len(predicted_labels)):
        if predicted_labels[idx] == target_labels[idx][0]:
            S_attacked_nodes.append(1)
        else:
            S_attacked_nodes.append(0)
    
    return succ, len(attack_nodes), nei_acc, len(neigh_nodes), D_attacked_nodes, S_attacked_nodes
