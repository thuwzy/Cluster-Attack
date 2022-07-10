import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
import torch.nn.functional as F

class SupervisedGraphSage(nn.Module):
    '''
        Traditional implementation of GCN needs matrix muliplying over the whole adjacent matrix.
        However, in our graph adversarial attack, GCN has to be evaluated under frequent changement of the graph.
        To allow for frequent changement of the graph, our GCN implemetation refers to GraphSAGE's format.
    '''

    def __init__(self, num_classes, num_hidden = 40, num_feature = 1433):
        super(SupervisedGraphSage, self).__init__()
        self.xent = nn.CrossEntropyLoss()
        
        self.agg1 = MeanAggregator()
        self.enc1 = Encoder(num_feature, 128, self.agg1)
        self.agg2 = MeanAggregator()
        self.enc2 = Encoder(self.enc1.embed_dim, num_classes, self.agg2)
              
    def forward(self, nodes, adj_lists, features):
        features1 = lambda nodes: F.relu(self.enc1(nodes, adj_lists, features)) # feature is a function!
        features2 = lambda nodes: self.enc2(nodes, adj_lists, features1)
        embeds = features2(nodes)
        return embeds        

    def loss(self, nodes, labels, adj_lists, features):
        scores = self.forward(nodes, adj_lists, features)
        return self.xent(scores, labels.squeeze())
