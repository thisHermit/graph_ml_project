import torch_geometric as pyg
import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

def load_peptide_func_dataset(batch_size=32):
    dataset = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func")

    # Check if dataset has splits; if not, create them manually
    if hasattr(dataset, 'train_val_test_idx'):
        peptides_train = dataset[dataset.train_val_test_idx['train']]
        peptides_val = dataset[dataset.train_val_test_idx['val']]
        peptides_test = dataset[dataset.train_val_test_idx['test']]
    else:
        # Create train, val, test splits manually
        num_train = int(0.8 * len(dataset))
        num_val = int(0.1 * len(dataset))
        num_test = len(dataset) - num_train - num_val
        peptides_train, peptides_val, peptides_test = torch.utils.data.random_split(
            dataset, [num_train, num_val, num_test]
        )

    train_loader = pyg.loader.DataLoader(peptides_train, batch_size=batch_size, shuffle=True)
    val_loader = pyg.loader.DataLoader(peptides_val, batch_size=batch_size, shuffle=False)
    test_loader = pyg.loader.DataLoader(peptides_test, batch_size=batch_size, shuffle=False)

    # Check number of classes and label distribution
    if hasattr(dataset, 'num_tasks'):
        num_classes = dataset.num_tasks
    elif hasattr(dataset, 'num_classes'):
        num_classes = dataset.num_classes
    else:
        # Assume binary classification if not specified
        num_classes = 1
    print(f"Number of classes (tasks): {num_classes}")

    all_labels = np.concatenate([data.y.numpy() for data in dataset], axis=0)
    label_distribution = np.mean(all_labels, axis=0)
    print(f"Label distribution: {label_distribution}")

    # Retrieve the number of node features
    # (Some PyG datasets use dataset.num_node_features or
    #  you may inspect the first data.x.size(1) for the correct dimension.)
    num_node_features = dataset.num_node_features if hasattr(dataset, 'num_node_features') \
                    else peptides_train[0].x.size(-1)
    print(f"Number of node features: {num_node_features}")

    return train_loader, val_loader, test_loader, num_node_features, num_classes


def load_mol_pcba_dataset(batch_size=32):
    dataset = PygGraphPropPredDataset(name = "ogbg-molpcba") 

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

    # Check number of classes and label distribution
    if hasattr(dataset, 'num_tasks'):
        num_classes = dataset.num_tasks
    elif hasattr(dataset, 'num_classes'):
        num_classes = dataset.num_classes
    else:
        # Assume binary classification if not specified
        num_classes = 1
    print(f"Number of classes (tasks): {num_classes}")

    all_labels = np.concatenate([data.y.numpy() for data in dataset], axis=0)
    label_distribution = np.mean(all_labels, axis=0)
    print(f"Label distribution: {label_distribution}")

    # Retrieve the number of node features
    # (Some PyG datasets use dataset.num_node_features or
    #  you may inspect the first data.x.size(1) for the correct dimension.)
    num_node_features = dataset.num_node_features if hasattr(dataset, 'num_node_features') \
                    else peptides_train[0].x.size(-1)
    print(f"Number of node features: {num_node_features}")

    return train_loader, val_loader, test_loader, num_node_features, num_classes