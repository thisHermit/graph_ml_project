import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric as pyg
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from torch.optim import AdamW
import copy
import statistics
import os

from torch_geometric.nn import TransformerConv, GCNConv, global_mean_pool


# find device
if torch.cuda.is_available(): # NVIDIA
    device = torch.device('cuda')
elif torch.backends.mps.is_available(): # apple M1/M2
    device = torch.device('mps')
else:
    device = torch.device('cpu')
device

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ----------------------------
# 1. Define the Graph GCN Model
# ----------------------------
class StudentModel(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=64, n_layers=2, dropout=0.5):
        """
        A GCN with a variable number of layers.

        :param in_channels:    Number of input features per node
        :param out_channels:   Number of prediction classes (or tasks)
        :param hidden_channels: Hidden layer size for the GCN layers
        :param n_layers:       Number of GCN layers
        :param dropout:        Dropout probability
        """
        super(StudentModel, self).__init__()

        # Create a list of GCN layers
        # First layer: in_channels -> hidden_channels
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Additional (n_layers - 1) layers: hidden_channels -> hidden_channels
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Final linear layer for graph-level prediction
        self.lin = nn.Linear(hidden_channels, out_channels)

        # Store dropout probability
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GCN model.
        
        :param x:         Node features, shape [num_nodes, in_channels]
        :param edge_index: Edge indices, shape [2, num_edges]
        :param batch:     Batch indices for each node, shape [num_nodes]
        :return:          Model output (logits), shape [batch_size, out_channels]
        """
        # Pass through each GCN layer
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Final classification/regression layer
        x = self.lin(x)
        return x


def baseline():
    # ----------------------------
    # 2. Prepare your dataset & loaders (same as your snippet)
    # ----------------------------

    dataset = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func")

    if hasattr(dataset, 'train_val_test_idx'):
        peptides_train = dataset[dataset.train_val_test_idx['train']]
        peptides_val = dataset[dataset.train_val_test_idx['val']]
        peptides_test = dataset[dataset.train_val_test_idx['test']]
    else:
        num_train = int(0.8 * len(dataset))
        num_val = int(0.1 * len(dataset))
        num_test = len(dataset) - num_train - num_val
        peptides_train, peptides_val, peptides_test = torch.utils.data.random_split(
            dataset, [num_train, num_val, num_test]
        )

    num_epochs = 30
    batch_size = 32
    train_loader = pyg.loader.DataLoader(peptides_train, batch_size=batch_size, shuffle=True)
    val_loader = pyg.loader.DataLoader(peptides_val, batch_size=batch_size, shuffle=False)
    test_loader = pyg.loader.DataLoader(peptides_test, batch_size=batch_size, shuffle=False)

    # Check number of classes (tasks)
    if hasattr(dataset, 'num_tasks'):
        num_classes = dataset.num_tasks
    elif hasattr(dataset, 'num_classes'):
        num_classes = dataset.num_classes
    else:
        num_classes = 1

    print(f"Number of classes (tasks): {num_classes}")

    all_labels = np.concatenate([data.y.numpy() for data in dataset], axis=0)
    label_distribution = np.mean(all_labels, axis=0)
    print(f"Label distribution: {label_distribution}")

    num_node_features = (
        dataset.num_node_features if hasattr(dataset, 'num_node_features')
        else peptides_train[0].x.size(-1)
    )
    print(f"Number of node features: {num_node_features}")

    metrics = {
        'test_loss': []
    }
    model_reference = None

    for seed in [42, 404, 64]:
        set_seed(seed)

        # ----------------------------
        # 3. Instantiate the GCN model, optimizer, and loss function
        # ----------------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = StudentModel(
            in_channels=num_node_features,
            out_channels=num_classes,
            hidden_channels=64,
            n_layers=4,
            dropout=0.5
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()  # For multi-task classification (Peptides-func)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # ----------------------------
        # 4. Define Training and Evaluation Routines
        # ----------------------------
        def train_one_epoch(loader):
            model.train()
            total_loss = 0
            for data in loader:
                data = data.to(device)
                data.x = data.x.float()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        @torch.no_grad()
        def evaluate(loader):
            model.eval()
            total_loss = 0
            for data in loader:
                data = data.to(device)
                data.x = data.x.float()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.float())
                total_loss += loss.item()
            return total_loss / len(loader)

        # ----------------------------
        # 5. Training Loop
        # ----------------------------
        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(train_loader)
            val_loss = evaluate(val_loader)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model_gcn.pt")
            
            print(f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}")

        # ----------------------------
        # 6. Testing the Best Model
        # ----------------------------
        model.load_state_dict(torch.load("best_model_gcn.pt"))
        test_loss = evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        metrics['test_loss'].append(test_loss)
        model_reference = model
    
    print("Model architecture:\n" + "=" * 20)
    print(model_reference)

    print("Metrics Reporting:")
    for metric, metrics_list in metrics:
        mean = statistics.mean(metrics_list)
        std_dev = statistics.stdev(metrics_list)
        print(f"{metric}: {mean} ± {std_dev}")


if __name__ == '__main__':
    baseline()