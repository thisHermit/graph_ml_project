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

from torch_geometric.nn import TransformerConv, GCNConv, global_mean_pool

from graph_ml_project.student.student import StudentModel
from graph_ml_project.utils.dataset import load_peptide_func_dataset, load_mol_pcba_dataset

# find device
if torch.cuda.is_available(): # NVIDIA
    device = torch.device('cuda')
elif torch.backends.mps.is_available(): # apple M1/M2
    device = torch.device('mps')
else:
    device = torch.device('cpu')
device

# ----------------------------
# 1. Define the Graph Transformer Model
# ----------------------------
class GraphTransformerModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        """
        A simple 2-layer Transformer-based GNN.

        :param in_channels:  Number of input features per node
        :param hidden_channels: Hidden layer size
        :param out_channels:   Number of prediction classes (or tasks)
        :param heads:          Number of attention heads in TransformerConv
        :param dropout:        Dropout probability
        """
        super(GraphTransformerModel, self).__init__()
        # First transformer layer
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # After conv1, the output dimension is hidden_channels * heads
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        # Final linear layer for graph-level prediction
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the model.
        :param x:         Node features [num_nodes, in_channels]
        :param edge_index: Edge indices [2, num_edges]
        :param batch:     Batch indices for each node [num_nodes]
        :return:          Model output (logits), shape [batch_size, out_channels]
        """
        # First transformer block
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second transformer block
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global average pooling over the nodes in each graph
        x = global_mean_pool(x, batch)
        
        # Optionally apply dropout before the final linear layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification/regression head
        x = self.lin(x)
        return x

def main(filename = "best_model1.pt"):
    # ----------------------------
    # 2. Prepare your dataset & loaders (from your snippet)
    # ----------------------------

    train_loader, val_loader, test_loader, num_node_features, num_classes = load_mol_pcba_dataset(batch_size=32)

    # ----------------------------
    # 3. Instantiate the model, optimizer, and loss function
    # ----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GraphTransformerModel(
        in_channels=num_node_features,
        hidden_channels=64,
        out_channels=num_classes,  # multi-task or single-task
        heads=4,
        dropout=0.5
    ).to(device)

    student_model = StudentModel(
        in_channels=num_node_features,
        out_channels=num_classes,
        hidden_channels=64,
        n_layers=4,
        dropout=0.5
    ).to(device)

    # If this is a multi-label classification problem, use BCEWithLogitsLoss
    # If single-label multi-class, use CrossEntropyLoss
    # If regression, use MSELoss, etc.
    # The Peptides-func dataset is typically multi-task classification, so BCEWithLogitsLoss is common:
    criterion = nn.BCEWithLogitsLoss()

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
            
            # data.y could have shape [batch_size, num_classes] for multi-label
            # Make sure to cast to float for BCEWithLogitsLoss
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
    num_epochs = 30

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(train_loader)
        val_loss = evaluate(val_loader)
        
        # Checkpoint if validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            torch.save(model.state_dict(), filename)
        
        print(f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}")

    # ----------------------------
    # 6. Testing the Best Model
    # ----------------------------
    model.load_state_dict(torch.load(filename))
    test_loss = evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    kd(model, student_model)


def kd(teacher_model, student_model):
    print("Teacher Model:")
    print(teacher_model)

    print("Student Model:")
    print(student_model)

    print("Do difficult stuff...")

    print("Distilled model!")


if __name__ == '__main__':
    main()