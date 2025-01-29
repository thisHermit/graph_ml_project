import torch
from torch_geometric.transforms import BaseTransform, Compose

import networkx as nx
import sys
sys.path.append('../')
from torch_scatter import scatter_add
import community as community_louvain
from model.data_utils import KHopTransform, PerformLouvainClustering, CustomLaplacianEigenvectorPE_B


import torch
import numpy as np
import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import AddLaplacianEigenvectorPE


class CustomLaplacianEigenvectorPE_B:
    def __init__(self, attr_name=None, start=20):
        self.attr_name = attr_name
        self.start = start  # Target PE dimension

    def __call__(self, data):
        original_x = data.x
        device = original_x.device
        pe_attr = self.attr_name or 'laplacian_eigenvector_pe'
        
        # Initialize PE tensor with zeros (shape: [num_nodes, start])
        pe = torch.zeros(data.num_nodes, self.start, device=device)

        for k in [self.start, 10, 5]:
            if k >= data.num_nodes - 1:
                continue  # Skip invalid k values

            try:
                # Compute PE and fill valid dimensions
                temp_data = data.clone()  # Lightweight copy
                transform = AddLaplacianEigenvectorPE(k=k, attr_name=self.attr_name)
                temp_data = transform(temp_data)
                
                temp_pe = getattr(temp_data, pe_attr)
                valid_dims = min(temp_pe.size(1), self.start)
                pe[:, :valid_dims] = temp_pe[:, :valid_dims]  # Fill valid dimensions
                break  # Exit on success

            except Exception as e:
                continue  # Try smaller k

        # Concatenate with original features
        data.x = torch.cat([original_x, pe], dim=1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(start={self.start})'



class PerformLouvainClustering(BaseTransform):
    def __call__(self, data):
        G = to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True)

        # Perform Louvain clustering
        partition = community_louvain.best_partition(G)
        # Convert partition (a dict) to a list of cluster labels
        labels = list(partition.values())
        data.louvain_cluster_id = torch.tensor(labels).view(-1, 1)
        return data

class AddGraphIdTransform:
    """
    Adds a global 'graph_id' and a global 'node_idx' to keep track of node ordering 
    so you can precisely map to teacher node embeddings.
    """
    def __init__(self):
        self.graph_id = 0
        self.node_count = 0

    def __call__(self, data):
        data.graph_id = self.graph_id
        self.graph_id += 1

        data.node_idx = torch.arange(self.node_count, self.node_count + data.num_nodes)
        self.node_count += data.num_nodes

        return data
    
    
class KHopTransform(BaseTransform):
    def __init__(self, K=[1, 2],agg='sum'):
        super(KHopTransform, self).__init__()
        self.K = K
        self.agg=agg

    def __call__(self, data):
        device = data.edge_index.device
        N = data.num_nodes
        edge_index = data.edge_index
        
        hop_features = []
        for k in self.K:
            # Create adjacency matrix A and raise it to the power of k
            A_k = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), size=(N, N))
            A_k = torch.sparse.mm(A_k, A_k) if k == 2 else A_k  # if k > 2, need to multiply more times
            
            D_k = A_k.to_dense().sum(dim=1)
            D_k_inv = torch.diag(D_k.pow(-1))
            D_k_inv = torch.where(torch.isinf(D_k_inv), torch.tensor(0.0, device=device, dtype=torch.float32), D_k_inv)
            hop_feature = torch.mm(A_k.to_dense(), D_k_inv @ data.x.float())
            hop_features.append(hop_feature)

        data.hop1_feature = hop_features[0]
        if len(self.K)==2:
            data.hop2_feature = hop_features[1]
        if "edge_attr" in data and self.agg=='sum':
                edge_feature = scatter_add(data.edge_attr, data.edge_index[1], dim=0, dim_size=data.num_nodes)
                data.edge_features = edge_feature.float()
        return data


# class RandomPathTransform(BaseTransform):
#     """
#     Uses the MuGSI-like approach:
#       - call nx.generate_random_paths(G, sample_size, path_length)
#       - store them in data.random_walk_paths
#     """
#     def __init__(self, sample_size=20, path_length=15):
#         super().__init__()
#         self.sample_size = sample_size
#         self.path_length = path_length

#     def __call__(self, data):
#         G = to_networkx(data, to_undirected=True)
#         try:
#             random_paths = nx.generate_random_paths(G, self.sample_size, self.path_length)
#             # This returns an iterable of lists. Convert it to a list and then a torch tensor
#             random_paths_list = list(random_paths)  # shape ~ [sample_size, path_length+1]
#             data.random_walk_paths = torch.tensor(random_paths_list, dtype=torch.long)
#         except:
#             # Fallback if something fails or no valid walks
#             data.random_walk_paths = torch.ones((self.sample_size, self.path_length+1), dtype=torch.long)

#         return data

class RandomPathTransform(BaseTransform):
    """
    Generates random walks for each single graph 'Data' object
    in shape [sample_size, path_length+1].
    """
    def __init__(self, sample_size=20, path_length=15):
        super().__init__()
        self.sample_size = sample_size
        self.path_length = path_length

    def __call__(self, data):
        G = to_networkx(data, to_undirected=True)
        paths = []
        try:
            nodes = list(G.nodes())
            for _ in range(self.sample_size):
                start_node = np.random.choice(nodes)
                walk = [start_node]
                for __ in range(self.path_length):
                    neighbors = list(G[walk[-1]])
                    if len(neighbors) == 0:
                        walk.append(walk[-1])
                    else:
                        walk.append(np.random.choice(neighbors))
                paths.append(walk)
            data.random_walk_paths = torch.tensor(paths, dtype=torch.long)
        except:
            data.random_walk_paths = torch.ones(
                (self.sample_size, self.path_length+1), dtype=torch.long
            )
        return data




class MolPCBA_Transform(BaseTransform):
    """
    Compose a few transforms used by MuGSI for a 1-hop GA-MLP baseline on ogbg-molpcba.
    If you want 2-hop, pass K=[1,2], etc.
    If you do not want clustering, set use_louvain=False.
    """
    def __init__(self, use_khop=True, use_louvain=False, use_lappe=True, khop_list=[1]):
        super().__init__()
        transforms_list = []
        # Apply Louvain Clustering first if enabled
        if use_louvain:
            transforms_list.append(PerformLouvainClustering())

        # Laplacian PE:
        if use_lappe:
            transforms_list.append(CustomLaplacianEigenvectorPE_B(start=20))
            
        # KHop: applied at the end
        if use_khop:
            # KHopTransform normally sets data.hop1_feature (and data.hop2_feature if K=[1,2], etc.)
            # We'll do just 1-hop:
            transforms_list.append(KHopTransform(K=khop_list, agg='sum')) 
            
        # if useRandomWalkConsistency:
            # transforms_list.append(RandomPathTransform())       

        self.compose = Compose(transforms_list)

    def __call__(self, data):
        data =  self.compose(data)
        # if data.y.dim() == 2 and data.y.size(0) == 1:
            # data.y = data.y.squeeze(0)
        data.y = data.y.float()
        data.x = data.x.float()
        return data