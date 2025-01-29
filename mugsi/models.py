import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, AttentionalAggregation
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN


class MLP(nn.Module):
    """
    The custom MLP model you showed. 
    If output_emb=True, returns: (logits, node_emb, graph_emb)
    If output_emb=False, returns: (logits)
    """
    def __init__(self, node_dim, hidden_dim, num_classes, pooling_method='sum'):
        super(MLP, self).__init__()
        self.node_dim = node_dim
        self.layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            BN(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            BN(2 * hidden_dim),
            nn.ReLU(),
        )
        self.pred_layer = nn.Linear(2 * hidden_dim, num_classes)

        if pooling_method == 'sum':
            self.pool = global_add_pool
        elif pooling_method == 'attention':
            self.pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, 1),
                )
            )
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

    def forward(self, data, output_emb=False):
        """
        data.x: node features, shape = [num_nodes, node_dim or more]
        data.batch: batch vector, shape = [num_nodes]
        If output_emb=True, return logits, node_emb, graph_emb
        """
        x = data.x[:, :self.node_dim].float()  # take only the first node_dim features
        batch = data.batch

        # node-wise transformations
        h = self.layers(x)  # shape [N, 2*hidden_dim]

        # graph-level pooling
        g = self.pool(h, batch)  # shape [num_graphs_in_batch, 2*hidden_dim]

        # final logit
        logits = self.pred_layer(g)  # shape [num_graphs_in_batch, num_classes]

        if output_emb:
            # h = node embeddings
            # g = pooled graph embeddings
            return logits, h, g
        else:
            return logits


class GA_MLP(nn.Module):
    def __init__(self, node_dim,edge_dim, hidden_dim, num_classes,use_edge_feats=True,pooling_method='sum',num_hops=1,**kargs):
        super(GA_MLP, self).__init__()
        self.node_dim = node_dim
        self.use_edge = use_edge_feats
        self.K = num_hops
        self.layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU())
        self.hop1_layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU())
        
        self.hop2_layers = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            BN(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            BN(2*hidden_dim),
            nn.ReLU())
        if use_edge_feats:
            self.edge_layers = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                BN(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2*hidden_dim),
                BN(2*hidden_dim),
                nn.ReLU(),
                nn.Linear(2*hidden_dim, 2*hidden_dim),
                BN(2*hidden_dim),
                nn.ReLU())
            
        self.pred_layer = nn.Linear(2*hidden_dim, num_classes)
        if pooling_method=='sum':
            self.pool = global_add_pool
        if pooling_method=='attention':
            # self.pool = AttentionalAggregation(gate_nn=nn.Linear(2*hidden_dim, 1))
            self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                                                     nn.ReLU(),
                                                                     nn.BatchNorm1d(hidden_dim),
                                                                     nn.Linear(hidden_dim, 1)))

    def forward(self,data,output_emb = False,testSpeed=False):
        x = data.x[:,:self.node_dim]
        hop1 = data.hop1_feature[:,:self.node_dim]
        batch = data.batch
        if not output_emb:
            x = self.layers(x)
            h1 = self.hop1_layers(hop1)
            h = x+h1
            if self.use_edge and "edge_features" in data:
                e = self.edge_layers(data.edge_features)
                h = h + e
            if testSpeed:
                return self.pool(h,data.batch)
            return self.pred_layer(self.pool(h,data.batch))
        else:
            # also output node emb for KD
            x = self.layers(x)
            h1 = self.hop1_layers(hop1)
            h = x+h1
            if self.use_edge and "edge_features" in data:
                e = self.edge_layers(data.edge_features)
                h = h + e
            g = self.pool(h,data.batch)
            return self.pred_layer(g),h,g # g is the pooled graph embeddings