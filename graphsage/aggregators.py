import torch
import torch.nn as nn
from torch.autograd import Variable
from math import sqrt

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using neighbors' embeddings
    """
    def __init__(self): 
        """
        Initializes the aggregator for a specific graph.
        """
        super(MeanAggregator, self).__init__()

    def forward(self, nodes, to_neighs, features, adj_lists):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        """
        # Local pointers to functions (speed hack)
        _set = set
        samp_neighs = to_neighs

        samp_neighs = [_set.union(samp_neigh, _set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} # nodes after adding neighbor nodes. Mapping from real idx to normalized idx.
        unique_nodes_r = {int(n):i for i,n in enumerate(nodes)} # nodes before adding neighbor nodes. Mapping from real idx to normalized idx.
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))

        # Normalization in GCN
        for i in nodes: # i, j are "real" indexes
            for j in samp_neighs[unique_nodes_r[int(i)]]:
                mask[unique_nodes_r[int(i)], unique_nodes[j]] = 1.0/sqrt((1.0+len(adj_lists[int(i)]))*(1.0+len(adj_lists[int(j)])))
        
        embed_matrix = features(unique_nodes_list)
        to_feats = mask.mm(embed_matrix)

        return to_feats
