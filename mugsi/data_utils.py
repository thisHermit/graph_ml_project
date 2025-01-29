import os
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.transforms import Compose

from molpcba_transforms import AddGraphIdTransform, RandomPathTransform

def load_ogb_molpcba_dataset(root, sample_size=20, path_length=15):
    """
    Loads the ogbg-molpcba dataset with custom transforms for 
    graph_id, node indexing, and random-walk paths.
    """
    
    _transform = AddGraphIdTransform()
    transforms = Compose( [
            RandomPathTransform(sample_size=sample_size, path_length=path_length)
    ])
    dataset = PygGraphPropPredDataset(
        name="ogbg-molpcba",
        root=root,
        pre_transform=_transform,
        transform=transforms  
    )
    
    
    split_idx = dataset.get_idx_split()
    train_set = dataset[split_idx["train"]]
    valid_set = dataset[split_idx["valid"]]
    test_set  = dataset[split_idx["test"]]

    return train_set, valid_set, test_set

def load_teacher_knowledge(teacher_kd_path, device):
    """
    teacher_kd_path is a .pt / .tar file containing:
      {
        "ptr": torch.LongTensor, # shape = [num_graphs+1]
        "logits": torch.FloatTensor, # [num_graphs, num_tasks]
        "h-embeddings": torch.FloatTensor, # [num_total_nodes, h_dim]
        "g-embeddings": torch.FloatTensor, # [num_graphs, h_dim]
      }
    """
    assert os.path.isfile(teacher_kd_path), f"No teacher knowledge at {teacher_kd_path}"
    knowledge = torch.load(teacher_kd_path, map_location=device)
    return (
        knowledge["ptr"].to(device),
        knowledge["logits"].float().to(device),
        knowledge["node_embeddings"].to(device),
        knowledge["graph_embeddings"].to(device),
    )
