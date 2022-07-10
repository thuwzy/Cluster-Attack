import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

from graphsage.graphsage import SupervisedGraphSage
from collections import defaultdict
import numpy as np
import random
import time
from sklearn.metrics import f1_score, accuracy_score

import scipy.sparse as sp

def load_cora(num_fake_nodes):
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes + num_fake_nodes, num_feats)) # A big feature matrix
    labels = np.empty((num_nodes + num_fake_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i # mapping the index of paper into [0, number of papers)
            if not info[-1] in label_map: # info[-1] is the label of paper
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set) # Adjacent Matrix
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    for i in range(num_nodes, num_nodes+num_fake_nodes): # initialize the fake nodes
        feat_data[i,:] = list(map(float, np.random.randint(2, size=num_feats))) # randomly initialize the feature of fake nodes
        labels[i] = random.randint(0, len(label_map)-1) # randomly initialize the label of fake nodes
    
    return feat_data, labels, adj_lists, num_nodes, num_feats, len(label_map)

def run_cora(features, labels, adj_lists, num_nodes, num_fake_nodes, num_feats):

    graphsage = SupervisedGraphSage(num_classes= 7, num_feature= 1433) # the model under attack
    rand_indices = np.random.permutation(num_nodes + num_fake_nodes) # randomly shuffle
    train = list(rand_indices[:270])
    val = rand_indices[270:810]
    test = rand_indices[810:] 
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    # optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), weight_decay = 1e-3)

    print("Start the training of victim model.")
    # start to train the model
    times = []
    for _ in range(100):
        random.shuffle(train) # shuffle training set
        batch_nodes = train[:256] # batch_size = 256 
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])), adj_lists, features)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        #print(batch, loss.data.item())

    val_output = graphsage.forward(val, adj_lists, features) 
    print("Finished the training of victim model.")
    print("Validation F1:", f1_score(labels[val], val_output.data.detach().cpu().numpy().argmax(axis=1), average="micro"))
    print("Validation Accuracy:", accuracy_score(labels[val], val_output.data.detach().cpu().numpy().argmax(axis=1)))
    print("Average batch time:", np.mean(times))

    return graphsage

def load_citeseer(num_fake_nodes):
    num_nodes = 3312
    num_feats = 3703
    feat_data = np.zeros((num_nodes + num_fake_nodes, num_feats)) # A big feature matrix
    labels = np.empty((num_nodes + num_fake_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("citeseer/citeseer.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i # mapping the index of paper into [0, number of papers)
            if not info[-1] in label_map: # info[-1] is the label of paper
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set) # Adjacent matrix
    cnt_no = 0
    with open("citeseer/citeseer.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            if (info[0] not in node_map.keys()) or (info[1] not in node_map.keys()):
                cnt_no += 1
                continue
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    print("Totally omit %d dangling edges." %(cnt_no))

    for i in range(num_nodes, num_nodes+num_fake_nodes): # Randomly initialize fake nodes
        feat_data[i,:] = list(map(float, np.random.randint(2, size=num_feats)))
        labels[i] = random.randint(0, len(label_map)-1) 
    
    return feat_data, labels, adj_lists, num_nodes, num_feats, len(label_map)

def run_citeseer(features, labels, adj_lists, num_nodes, num_fake_nodes, num_feats):

    graphsage = SupervisedGraphSage(num_classes = 6, num_feature = 3703) # the model under attack
    rand_indices = np.random.permutation(num_nodes + num_fake_nodes) # shuffle
    train = list(rand_indices[:330])
    val = rand_indices[330:990]
    test = rand_indices[990:]
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    # optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), weight_decay = 1e-3)

    print("Start the training of victim model.")
    # start to train the model
    times = []
    for _ in range(100):
        random.shuffle(train) # shuffle
        batch_nodes = train[:256] # batch_size = 256
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])), adj_lists, features)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        #print(batch, loss.data.item())

    val_output = graphsage.forward(val, adj_lists, features) 
    print("Finished the training of victim model.")
    print("Validation F1:", f1_score(labels[val], val_output.data.detach().cpu().numpy().argmax(axis=1), average="micro"))
    print("Validation Accuracy:", accuracy_score(labels[val], val_output.data.detach().cpu().numpy().argmax(axis=1)))
    print("Average batch time:", np.mean(times))

    return graphsage

import os
import pickle as pkl
import scipy.sparse as sp
def loadogb(st,dir):
    process_dir=dir+"/processed"
    
    wi=os.listdir(process_dir)
    # signifies that the data is processed
    if 'adj.pkl' in wi:
        adj=pkl.load(open(process_dir+'/adj.pkl','rb'))
        feat=np.load(process_dir+'/feature.npy')
        label=np.load(process_dir+'/labels.npy')
        train_index=np.load(process_dir+'/train_index.npy')
        val_index=np.load(process_dir+'/val_index.npy')
        test_index=np.load(process_dir+'/test_index.npy')
        return adj,feat,label,train_index,val_index,test_index
    else:
        path=dir+"/raw"
        wt=[]
        feats=open(path+"/node-feat.csv")
        ff=feats.readlines()
        for i in range(len(ff)):
            spc=ff[i].split(',')
            
            nlist=list(map(lambda j:float(j),spc))
            wt.append(nlist)

        feat=np.array(wt)
        num_nodes=len(feat)
        feats.close()
        np.save(process_dir+'/feature.npy',feat)
        lbs=open(path+"/node-label.csv")
        lb=lbs.readlines()
        ll=[]
        for i in range(len(lb)):
            ll.append(int(lb[i]))
        labels=np.array(ll)
        np.save(process_dir+'/labels.npy',labels)
        lbs.close()
        edges=open(path+"/edge.csv")
        eds=sp.lil_matrix((num_nodes,num_nodes))
        edge=edges.readlines()
        nownum=0
        for line in edge:
            nownum+=1
            if nownum%200000==0:
                print(nownum)
            id1=int(line.split(',')[0])
            id2=int(line.split(',')[1])
            eds[id1,id2]=1
            eds[id2,id1]=1
        eds=eds.tocsr()
        pkl.dump(eds,open(process_dir+"/adj.pkl","wb+"))
        edges.close()
        if st=='arxiv':
            sp_path=dir+"/split/time"
        if st=='products':
            sp_path=dir+"/split/sales_ranking"
        train_index=[]
        t=open(sp_path+"/train.csv").readlines()
        for line in t:
            train_index.append(int(line))
        tr_index=np.array(train_index)
        np.save(process_dir+'/train_index.npy',tr_index)
        valid_index=[]
        v=open(sp_path+"/valid.csv").readlines()
        for line in v:
            valid_index.append(int(line))
        val_index=np.array(valid_index)
        np.save(process_dir+'/val_index.npy',val_index)
        test_index=[]
        te=open(sp_path+"/test.csv").readlines()
        for line in te:
            test_index.append(int(line))
        te_index=np.array(test_index)
        np.save(process_dir+'/test_index.npy',te_index)
        return eds,feat,labels,train_index,val_index,te_index