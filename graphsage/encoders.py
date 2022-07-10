import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, feature_dim, 
            embed_dim, aggregator): 
        super(Encoder, self).__init__()

        self.feat_dim = feature_dim
        self.aggregator = aggregator

        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, adj_lists, features):
        """
        Generates embeddings for a batch of nodes.

        nodes    -- list of nodes
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        """
        neigh_feats = self.aggregator.forward(nodes, [adj_lists[int(node)] for node in nodes], features, adj_lists)
        combined = neigh_feats
        combined = self.weight.mm(combined.t())

        return combined.t()
