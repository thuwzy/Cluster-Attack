import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        
        x=self.ll(x)
        x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x

class GCN(nn.Module):
    def __init__(self,num_layers,num_features,activation=F.elu):
        super(GCN, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        
        for i in range(num_layers):
            
            if i!=num_layers-1:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1],activation=activation,dropout=True).cuda())
            else:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1]).cuda())
        
    def forward(self,x,adj,dropout=0):
        
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
       # x=F.softmax(x, dim=-1)
        return x

class gcn(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(gcn,self).__init__()
        #self.ln=nn.LayerNorm(in_feat)
        self.gcn=GCN(4,[in_feat,256,128,64,out_feat])
    def forward(self,x,adj,dropout=0):
        x=self.gcn(x,adj,dropout=dropout)
        return x

def GCNadj(adj,pow=-0.5):
    adj2=sp.eye(adj.shape[0])+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
    adj2 = sp.coo_matrix(adj2)
    
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=d_mat_inv_sqrt @ adj2 @ d_mat_inv_sqrt
    
    return adj2.tocoo()