#!/usr/bin/env python

import datetime
import os
import logging
import torch
import torch.nn as nn
import networkx as nx
import community as community_louvain

# GraphGPS & PyG imports
import graphgps  # noqa, registers custom modules
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, dump_cfg, set_cfg, load_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import nn as pyg_nn
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from graphgps.logger import create_logger
from typing import List, Dict, Callable, Iterable


###############################################################################
# 1) PER-GRAPH LOUVAIN CLUSTERING
###############################################################################
class PerGraphLouvainClustering(BaseTransform):
    """
    Perform Louvain community detection separately for each graph in the batch.
    Expects that each node has a 'batch' attribute telling us which sub-graph
    the node belongs to, and a 'graph_id' giving the global ID.
    """
    def __call__(self, data):
        # Single-graph scenario
        if data.num_graphs == 1:
            data.louvain_cluster_id = _run_louvain_on_single_graph(data)
            return data
        
        # Multi-graph (batched) scenario
        cluster_ids = torch.empty((data.num_nodes,), dtype=torch.long)
        
        for local_g_id in range(data.num_graphs):
            mask = (data.batch == local_g_id)
            sub_data = _extract_subgraph(data, mask)
            sub_cluster_id = _run_louvain_on_single_graph(sub_data)
            cluster_ids[mask] = sub_cluster_id
        
        data.louvain_cluster_id = cluster_ids.unsqueeze(-1)  # shape [num_nodes, 1]
        return data


def _run_louvain_on_single_graph(data):
    """Run Louvain on a single-graph Data object and return cluster IDs as a tensor."""
    G = to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True)
    if G.number_of_nodes() == 0:
        return torch.zeros((data.num_nodes,), dtype=torch.long)
    
    partition = community_louvain.best_partition(G, random_state=44)
    labels = list(partition.values())  # length = number_of_nodes
    return torch.tensor(labels, dtype=torch.long)


def _extract_subgraph(batch_data, mask):
    """Extract the subgraph corresponding to 'mask' from a batched Data object."""
    sub_data = batch_data.__class__()
    
    # Node features (anything of size [num_nodes, *])
    for key, val in batch_data:
        if isinstance(val, torch.Tensor) and val.size(0) == batch_data.num_nodes:
            sub_data[key] = val[mask]
        else:
            sub_data[key] = val
    
    # Subset edge_index
    if batch_data.edge_index is not None:
        edge_index = batch_data.edge_index
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        sub_data.edge_index = _reindex_edge_index(edge_index[:, edge_mask], mask)

    sub_data.num_nodes = mask.sum().item()
    return sub_data


def _reindex_edge_index(edge_index, mask):
    """Reindex a subgraph's edge_index so that node IDs start at 0."""
    idx = torch.nonzero(mask, as_tuple=True)[0]  # which node indices are included
    new_index = torch.empty_like(mask, dtype=torch.long)
    new_index[idx] = torch.arange(idx.size(0), dtype=torch.long)
    return new_index[edge_index]


###############################################################################
# 2) OPTIONAL RANDOM WALK TRANSFORM
###############################################################################
class RandomPathTransform(BaseTransform):
    """Generate random walks for path-based knowledge (if needed)."""
    def __init__(self, sample_size=20, path_length=15):
        super().__init__()
        self.sample_size = sample_size
        self.path_length = path_length

    def __call__(self, data):
        if data.num_graphs == 1:
            data.random_walk_paths = _random_walk_single_graph(data, self.sample_size, self.path_length)
            return data
        
        # Multi-graph batch
        all_paths = torch.ones((data.num_nodes, self.sample_size, self.path_length+1), dtype=torch.long)
        for local_g_id in range(data.num_graphs):
            mask = (data.batch == local_g_id)
            sub_data = _extract_subgraph(data, mask)
            sub_paths = _random_walk_single_graph(sub_data, self.sample_size, self.path_length)
            if sub_data.num_nodes > 0:
                node_idxs = torch.nonzero(mask, as_tuple=True)[0]
                for n_id in node_idxs:
                    all_paths[n_id] = sub_paths
        data.random_walk_paths = all_paths
        return data


def _random_walk_single_graph(data, sample_size, path_length):
    G = to_networkx(data, node_attrs=None, edge_attrs=None)
    if G.number_of_nodes() == 0:
        return torch.ones((sample_size, path_length+1), dtype=torch.long)
    
    paths = []
    for _ in range(sample_size):
        start_node = torch.randint(0, G.number_of_nodes(), (1,)).item()
        path = [start_node]
        current_node = start_node
        for _ in range(path_length):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            current_node = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
            path.append(current_node)
        while len(path) < path_length + 1:
            path.append(path[-1])
        paths.append(path)
    return torch.tensor(paths, dtype=torch.long)


###############################################################################
# 3) FORWARD HOOK FOR NODE EMBEDDINGS
###############################################################################
class EmbeddingsExtractor(nn.Module):
    """
    Wraps a GraphGPS model, hooking into a specified sub-layer to capture
    node-level outputs. We store them in self._features.
    """
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self._save_outputs_hook(layer_id))

    def _save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(module, input, output):
            self._features[layer_id] = output
        return fn

    def forward(self, batch):
        """
        1) Forward pass model -> triggers the hook(s).
        2) Return final model output (logits) + the captured node embeddings.
        """
        logits, _ = self.model(batch)
        return logits, {lid: self._features[lid] for lid in self.layers}


###############################################################################
# 4) SUB-GRAPH (CLUSTER) EMBEDDINGS
###############################################################################
def get_cluster_embeddings(node_embeddings: torch.Tensor,
                           cluster_ids: torch.Tensor,
                           pooling='mean'):
    """
    node_embeddings: shape [num_nodes_in_graph, hidden_dim]
    cluster_ids: shape [num_nodes_in_graph] (integer IDs)
    pooling: 'mean' or 'sum'
    Return: cluster_embs -> shape [num_clusters, hidden_dim]
            cluster_labels -> shape [num_clusters]
    """
    unique_clusters = torch.unique(cluster_ids)
    cluster_embs_list = []
    cluster_labels_list = []

    for c_id in unique_clusters:
        mask = (cluster_ids == c_id)
        if pooling == 'mean':
            c_emb = node_embeddings[mask].mean(dim=0, keepdim=True)
        else:
            c_emb = node_embeddings[mask].sum(dim=0, keepdim=True)
        cluster_embs_list.append(c_emb)
        cluster_labels_list.append(c_id.view(-1))

    cluster_embs = torch.cat(cluster_embs_list, dim=0)
    cluster_labels = torch.cat(cluster_labels_list, dim=0)
    return cluster_embs, cluster_labels


###############################################################################
# 5) MAIN EXTRACTION FUNCTION
###############################################################################
def get_embeddings(model: nn.Module, loaders: List, cfg):
    """
    Extract node embeddings (via forward hook), graph embeddings, sub-graph (Louvain)
    embeddings, logits, and store ptr, graph_id, node2graph as well.
    """
    device = torch.device(cfg.accelerator if cfg.accelerator else 'cpu')
    model.to(device)
    model.eval()

    # Compose transforms if needed
    # E.g. Louvain + random paths
    transforms = Compose([
        PerGraphLouvainClustering(),
        RandomPathTransform(path_length=cfg.get('path_length', 15)),
    ])

    # Prepare the forward hook for node embeddings from the desired layer
    hook_layer = 'model.layers.4.ff_dropout2'  # e.g. 'model.layers.4.ff_dropout2'
    extractor = EmbeddingsExtractor(model, [hook_layer]).to(device)

    # Prepare placeholders
    hidden_dim = int(cfg.gt.dim_hidden)
    num_graphs = cfg.dataset.num_graphs
    num_nodes = cfg.dataset.num_nodes
    num_tasks = cfg.dataset.num_tasks

    # Tensors for teacher knowledge
    node_embeddings = torch.zeros((num_nodes, hidden_dim), dtype=torch.float)
    graph_embeddings = torch.zeros((num_graphs, hidden_dim), dtype=torch.float)
    logits_embeddings = torch.zeros((num_graphs, num_tasks), dtype=torch.float)

    # 'ptr' offsets for slicing nodes
    ptr = torch.zeros((num_graphs+1,), dtype=torch.long)
    # We'll keep track of how many nodes have been assigned in each graph
    graph_node_count = torch.zeros((num_graphs,), dtype=torch.long)

    # We'll also store node -> graph for convenience (if not already in teacher knowledge)
    node2graph = torch.zeros((num_nodes,), dtype=torch.long)

    # We'll store cluster embeddings in a list
    cluster_embeddings_list = []
    
    ############################################################################
    # Rolling offset for node embeddings
    ############################################################################
    global_node_offset = 0  # how many nodes we've seen so far

    # We iterate over the data
    with torch.no_grad():
        for loader_idx, loader in enumerate(loaders):
            logging.info(f"[Embeddings] Loader {loader_idx+1}/{len(loaders)} start.")
            
            for batch_idx, batch in enumerate(loader):
                batch = transforms(batch)
                batch = batch.to(device)

                #  1) Forward pass, capture node embeddings
                batch_logits, hook_outputs = extractor(batch)
                h_embeddings = hook_outputs[hook_layer]  # node embeddings in this batch - # shape: [num_nodes_in_batch, hidden_dim]
                batch_logits = batch_logits.float()       # shape: [num_graphs_in_batch, num_tasks]
                
                num_nodes_in_batch = batch.num_nodes
                local_graph_ids = torch.unique(batch.batch)  # local graph IDs in [0..(num_graphs_in_batch-1)]
                
                 # 2) Place node embeddings in global tensor
                start_idx = global_node_offset
                end_idx = global_node_offset + num_nodes_in_batch
                node_embeddings[start_idx:end_idx] = h_embeddings.cpu()
                
                 # 3) Graph-level summary embeddings
                #    We'll do global mean pool for each sub-graph
                summary_batch = pyg_nn.global_mean_pool(h_embeddings, batch.batch).cpu()  # shape: [num_graphs_in_batch, hidden_dim]


                # 4) Map local -> global graph IDs, store summary, store logits
                for i, local_g in enumerate(local_graph_ids):
                    g_id = batch.graph_id[local_g].item()
                    graph_embeddings[g_id] = summary_batch[i]
                    logits_embeddings[g_id] = batch_logits[i].cpu()
                # 5) Sub-graph (Louvain) embeddings
                cluster_ids_all = batch.louvain_cluster_id.squeeze(-1)  # shape: [num_nodes_in_batch]
                for local_g in local_graph_ids:
                    mask = (batch.batch == local_g)
                    if mask.sum() == 0:
                        continue
                    g_id = batch.graph_id[local_g].item()
                    sub_node_emb = h_embeddings[mask]  # node embeddings for that local graph
                    sub_cluster_ids = cluster_ids_all[mask]  # cluster labels for those nodes
                    sub_cluster_embs, sub_cluster_labels = get_cluster_embeddings(sub_node_emb, sub_cluster_ids)

                    cluster_embeddings_list.append({
                        'graph_id': g_id,
                        'cluster_labels': sub_cluster_labels.cpu(),   # [C]
                        'cluster_embs': sub_cluster_embs.cpu()        # [C, hidden_dim]
                    })
                
                # 5) Sub-graph (Louvain) embeddings
                cluster_ids_all = batch.louvain_cluster_id.squeeze(-1)  # shape: [num_nodes_in_batch]
                for local_g in local_graph_ids:
                    mask = (batch.batch == local_g)
                    if mask.sum() == 0:
                        continue
                    g_id = batch.graph_id[local_g].item()
                    sub_node_emb = h_embeddings[mask]  # node embeddings for that local graph
                    sub_cluster_ids = cluster_ids_all[mask]  # cluster labels for those nodes
                    sub_cluster_embs, sub_cluster_labels = get_cluster_embeddings(sub_node_emb, sub_cluster_ids)

                    cluster_embeddings_list.append({
                        'graph_id': g_id,
                        'cluster_labels': sub_cluster_labels.cpu(),   # [C]
                        'cluster_embs': sub_cluster_embs.cpu()        # [C, hidden_dim]
                    })
                    
                # 6) Fill in node2graph and update graph_node_count for each local graph
                #    In PyG, batch.ptr is [num_graphs_in_batch+1], telling how many nodes
                #    belong to each local graph in the batch. So for each local graph g in 0..(num_graphs_in_batch-1):
                #    the node range is [ptr[g], ptr[g+1]) in the *batch*. We offset that globally.

                local_ptr = batch.ptr  # shape: [num_graphs_in_batch+1]
                for g_local in range(batch.num_graphs):
                    g_id = batch.graph_id[g_local].item()
                    sub_start = local_ptr[g_local].item()
                    sub_end = local_ptr[g_local+1].item()
                    sub_count = sub_end - sub_start
                    # The global node range is:
                    global_start = start_idx + sub_start
                    global_end = start_idx + sub_end

                    # Fill node2graph
                    node2graph[global_start:global_end] = g_id
                    # Accumulate the count for this global graph
                    graph_node_count[g_id] += sub_count

                global_node_offset += num_nodes_in_batch
                torch.cuda.empty_cache()

        # 7) Build the ptr array from graph_node_count
        for i in range(num_graphs):
            ptr[i+1] = ptr[i] + graph_node_count[i]


        # Build 'ptr' from graph_node_count
        # ptr[i+1] = ptr[i] + graph_node_count[i]
        ptr[0] = 0
        for i in range(num_graphs):
            ptr[i+1] = ptr[i] + graph_node_count[i]

   ############################################################################
    # Final dictionary: node emb, graph emb, cluster emb, logits, ptr, etc.
    ############################################################################
    teacher_knowledge = {
        "node_embeddings": node_embeddings,            # [num_nodes, hidden_dim]
        "graph_embeddings": graph_embeddings,          # [num_graphs, hidden_dim]
        "cluster_embeddings": cluster_embeddings_list, # list of dicts
        "logits": logits_embeddings,                   # [num_graphs, num_tasks]
        "ptr": ptr,                                    # [num_graphs+1]
        "node2graph": node2graph,                      # [num_nodes]
        "graph_node_count": graph_node_count,          # [num_graphs]
    }

    # Save
    save_path = os.path.join(cfg.run_dir, "teacher-knowledge.pt")
    torch.save(teacher_knowledge, save_path)
    logging.info(f"[Embeddings] Teacher knowledge saved to {save_path}")


###############################################################################
# 6) MAIN SCRIPT
###############################################################################
def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """
    Optional helper that modifies cfg.out_dir based on config filename + name tag,
    if you want to create a custom run directory. Adjust as you like.
    """
    import os
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    if name_tag:
        run_name += f"-{name_tag}"
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


if __name__ == "__main__":
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    # Optionally set custom out_dir
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)

    torch.set_num_threads(cfg.num_threads)
    set_printing()
    seed_everything(cfg.seed)
    auto_select_device()

    # If using pretrained checkpoint
    if cfg.pretrained.dir:
        cfg = load_pretrained_model_cfg(cfg)

    logging.info(f"[Embedding] Starting extraction: {datetime.datetime.now()}")
    logging.info(f"[Embedding] Type: {cfg.embeddings.type}")

    # Create loaders
    loaders = create_loader()
    # Create model
    model = create_model()
    # Create logger
    logger = create_logger()

    # Load pretrained if needed
    if cfg.pretrained.dir:
        model = init_model_from_pretrained(
            model,
            cfg.pretrained.dir,
            cfg.pretrained.freeze_main,
            cfg.pretrained.reset_prediction_head,
            seed=cfg.seed
        )

    # Extract embeddings
    get_embeddings(model, loaders, cfg)
