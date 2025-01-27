import torch
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric import nn as nng
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from copy import copy
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import argparse

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # MPS is currently slower than CPU due to missing int64 min/max ops
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Using device: {device}", flush=True)

base_dir = os.getenv(
    "BASE_DIR",
    f"/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GakD",
)

if not os.path.exists(base_dir):
    os.makedirs(base_dir, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for GINENetwork.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        network = [
            nn.Linear(in_dim, 2 * in_dim),
            nn.BatchNorm1d(2 * in_dim),
            nn.ReLU(),
            nn.Linear(2 * in_dim, out_dim),
        ]
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class OGBMolEmbedding(nn.Module):
    """
    OGBMolEmbedding class for embedding molecules using atom and bond encoders.
    """

    def __init__(self, dim):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=dim)
        self.bond_encoder = BondEncoder(emb_dim=dim)

    def forward(self, data):
        data = copy(data)
        data.x = self.atom_encoder(data.x)
        data.edge_attr = self.bond_encoder(data.edge_attr)
        return data


class VNAgg(nn.Module):
    """
    VNAgg class for aggregating virtual nodes using MLP. Includes trainable epsilon.
    """

    def __init__(self, dim, train_eps=False, eps=0.0):
        super().__init__()
        self.mlp = nn.Sequential(MLP(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.train_eps = train_eps
        self.eps = (
            nn.Parameter(torch.Tensor([eps])) if train_eps else torch.Tensor([eps])
        )

    def forward(self, virtual_node, embeddings, batch_idx):
        if batch_idx.size(0) > 0:
            sum_embeddings = nng.global_add_pool(embeddings, batch_idx)
        else:
            sum_embeddings = torch.zeros_like(virtual_node, device=device)
        virtual_node = (1 + self.eps.to(device)) * virtual_node.to(
            device
        ) + sum_embeddings.to(device)
        virtual_node = self.mlp(virtual_node)
        return virtual_node


class GlobalPool(nn.Module):
    """
    GlobalPool class for global pooling of node embeddings. Supports mean and sum pooling.

    """

    def __init__(self, fun):
        super().__init__()
        self.fun = getattr(nng, "global_{}_pool".format(fun.lower()))

    def forward(self, data):
        h, batch_idx = data.x, data.batch
        pooled = self.fun(h, batch_idx, size=data.num_graphs)
        return pooled


class ConvBlock(nn.Module):
    """
    GINE ConvBlock with given parameters. Uses GINEConv Layer from torch_geometric.
    """

    def __init__(
        self,
        dim,
        dropout=0.5,
        activation=F.relu,
        virtual_node=False,
        virtual_node_agg=True,
        last_layer=False,
        train_vn_eps=False,
        vn_eps=0.0,
    ):

        super().__init__()
        self.conv = nng.GINEConv(MLP(dim, dim), train_eps=True)
        self.bn = nn.BatchNorm1d(dim)
        self.activation = activation or nn.Identity()
        self.dropout_ratio = dropout
        self.last_layer = last_layer
        self.virtual_node = virtual_node
        self.virtual_node_agg = virtual_node_agg

        if self.virtual_node and self.virtual_node_agg:
            self.virtual_node_agg = VNAgg(dim, train_eps=train_vn_eps, eps=vn_eps)

    def forward(self, data):
        """
        Forward pass for GINE ConvBlock.
        """
        data = copy(data)
        h, edge_index, edge_attr, batch_idx = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # Add virtual node embeddings to node embeddings
        if self.virtual_node:
            h = h + data.virtual_node[batch_idx]
        # Apply GINE convolution
        h = self.conv(h, edge_index, edge_attr)
        # Apply batch normalization
        h = self.bn(h)
        # Apply activation function if not the last layer
        if not self.last_layer:
            h = self.activation(h)
        # Apply dropout
        h = F.dropout(h, self.dropout_ratio, training=self.training)
        # Aggregate virtual nodes if enabled
        if self.virtual_node and self.virtual_node_agg:
            v = self.virtual_node_agg(data.virtual_node, h, batch_idx)
            v = F.dropout(v, self.dropout_ratio, training=self.training)
            data.virtual_node = v
        # Update node embeddings
        data.x = h
        return data


class GINENetwork(nn.Module):
    """
    GINENetwork class using ConvBlocks to build the network.
    """

    def __init__(
        self,
        hidden_dim=100,
        out_dim=128,
        num_layers=3,
        dropout=0.5,
        virtual_node=False,
        train_vn_eps=False,
        vn_eps=0.0,
        return_embeddings=False,
    ):
        """
        Initialize GINENetwork with given parameters. Returns (logits,embeddings) if return_embeddings is True, else returns logits.
        """
        super().__init__()
        self.return_embeddings = return_embeddings
        convs = [
            ConvBlock(
                hidden_dim,
                dropout=dropout,
                virtual_node=virtual_node,
                train_vn_eps=train_vn_eps,
                vn_eps=vn_eps,
            )
            for _ in range(num_layers - 1)
        ]
        convs.append(
            ConvBlock(
                hidden_dim,
                dropout=dropout,
                virtual_node=virtual_node,
                virtual_node_agg=False,
                last_layer=True,
                train_vn_eps=train_vn_eps,
                vn_eps=vn_eps,
            )
        )
        self.network = nn.Sequential(OGBMolEmbedding(hidden_dim), *convs)
        # Aggregate embeddings using mean pooling and MLP
        self.aggregate = nn.Sequential(
            GlobalPool("mean"),
            MLP(hidden_dim, out_dim),
        )

        self.virtual_node = virtual_node
        if self.virtual_node:
            self.v0 = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True)

    def forward(self, data):
        """
        Forward pass for GINENetwork.
        """
        # Initialize virtual node embeddings if enabled
        if self.virtual_node:
            data.virtual_node = self.v0.expand(data.num_graphs, self.v0.shape[-1])
        H = self.network(data)
        # Return embeddings if enabled
        if self.return_embeddings:
            # logits, embeddings
            return self.aggregate(H), H.x
        # Return logits
        return self.aggregate(H)


class GINETrainer:
    """
    GINETrainer class for training GINENetwork.
    """

    def __init__(
        self,
        dataset_name="ogbg-molpcba",
        num_layers=5,
        hidden_dim=400,
        dropout=0.5,
        virtual_node=True,
        train_vn_eps=False,
        vn_eps=0.0,
        lr=0.001,
        batch_size=100,
        num_workers=4,
    ):
        # Initialize dataset
        self.dataset_name = dataset_name
        os.makedirs(f"{base_dir}/data", exist_ok=True)
        self.dataset = PygGraphPropPredDataset(
            name=dataset_name, root=f"{base_dir}/data"
        )
        self.split_idx = self.dataset.get_idx_split()

        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.dataset[self.split_idx["train"]],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.valid_loader = DataLoader(
            self.dataset[self.split_idx["valid"]],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            self.dataset[self.split_idx["test"]],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Initialize model
        self.model = GINENetwork(
            hidden_dim=hidden_dim,
            out_dim=self.dataset.num_tasks,
            num_layers=num_layers,
            dropout=dropout,
            virtual_node=virtual_node,
            train_vn_eps=train_vn_eps,
            vn_eps=vn_eps,
        ).to(device)

        # Initialize optimizer and criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize evaluator
        self.evaluator = Evaluator(name=self.dataset_name)
        # Initialize virtual node and epsilon parameters
        self.virtual_node = virtual_node
        self.train_vn_eps = train_vn_eps
        self.vn_eps = vn_eps

    def train(self, epochs=100):
        """
        Train the GINENetwork.
        """
        best_valid_ap = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            y_true_list = []
            y_pred_list = []

            for batch in self.train_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()

                y_pred = self.model(batch)
                y_true = batch.y.float()
                y_available = ~torch.isnan(y_true)

                loss = self.criterion(y_pred[y_available], y_true[y_available])
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                y_true_list.append(y_true.detach().cpu())
                y_pred_list.append(y_pred.detach().cpu())

            train_loss /= len(self.train_loader)

            if epoch % max(1, epochs // 10) == 0:
                # Evaluate on validation set
                valid_ap = self.eval(split="valid")
                print(
                    f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid AP: {valid_ap:.4f}",
                    flush=True,
                )

                # Save best model
                if valid_ap > best_valid_ap:
                    best_valid_ap = valid_ap
                    os.makedirs(f"{base_dir}/models", exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        f"{base_dir}/models/gine_model_{self.dataset_name}_virtual_node_{self.virtual_node}_train_vn_eps_{self.train_vn_eps}_vn_eps_{self.vn_eps}.pt",
                    )

    @torch.no_grad()
    def eval(self, split="valid"):
        """
        Evaluate the GINENetwork on the validation or test set.
        """
        self.model.eval()
        loader = self.valid_loader if split == "valid" else self.test_loader
        y_true_list = []
        y_pred_list = []

        for batch in loader:
            batch = batch.to(device)
            y_pred = self.model(batch)
            y_true = batch.y

            y_true_list.append(y_true.detach().cpu())
            y_pred_list.append(y_pred.detach().cpu())

        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_pred = torch.cat(y_pred_list, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        if self.dataset_name == "ogbg-molpcba":
            return self.evaluator.eval(input_dict)["ap"]
        else:
            return self.evaluator.eval(input_dict)["rocauc"]


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def run_multiple_experiments(
    dataset_name="ogbg-molpcba",
    n_runs=5,
    num_layers=5,
    hidden_dim=400,
    dropout=0.5,
    virtual_node=True,
    train_vn_eps=False,
    vn_eps=0.0,
    lr=0.001,
    batch_size=32,
    epochs=100,
    seed=42,
    output_file="experiment_results.csv",
):
    """
    Run multiple training experiments and save results to CSV

    Args:
        n_runs: Number of experimental runs
        num_layers: Number of GNN layers
        hidden_dim: Hidden dimension size
        dropout: Dropout rate
        virtual_node: Whether to use virtual node
        train_vn_eps: Whether to train virtual node epsilon
        vn_eps: Virtual node epsilon value
        lr: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        seed: Random seed
        output_file: Path to save results CSV
    """

    # Initialize results storage
    results = []

    # Create unique experiment ID using timestamp
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric = "ap" if dataset_name == "ogbg-molpcba" else "rocauc"
    for run in range(n_runs):
        print(f"\nStarting Run {run + 1}/{n_runs}", flush=True)

        # Set different seed for each run
        set_seed(seed + run)

        # Initialize trainer
        trainer = GINETrainer(
            dataset_name=dataset_name,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            virtual_node=virtual_node,
            train_vn_eps=train_vn_eps,
            vn_eps=vn_eps,
            lr=lr,
            batch_size=batch_size,
        )

        # Train model
        trainer.train(epochs=epochs)

        # Get final validation and test AP
        valid_ap = trainer.eval(split="valid")
        test_ap = trainer.eval(split="test")

        # Store results
        run_results = {
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "run": run + 1,
            "seed": seed + run,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "virtual_node": virtual_node,
            "train_vn_eps": train_vn_eps,
            "vn_eps": vn_eps,
            "n_params": numel(trainer.model, only_trainable=True),
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "valid_metric": valid_ap,
            "test_metric": test_ap,
            "metric": metric,
        }
        results.append(run_results)

        # Save intermediate results after each run
        df = pd.DataFrame([run_results])
        # check if output file exists
        if os.path.exists(output_file):
            with open(output_file, "a") as f:
                f.write("\n")
            df.to_csv(output_file, index=False, mode="a", header=False)
        else:
            df.to_csv(output_file, index=False)

        print(f"Run {run + 1} Results:", flush=True)
        print(f"Validation {metric}: {valid_ap:.4f}", flush=True)
        print(f"Test {metric}: {test_ap:.4f}", flush=True)

    # Calculate and print summary statistics
    df = pd.DataFrame(results)
    summary = df[["valid_metric", "test_metric"]].agg(["mean", "std"])

    print("\nSummary Statistics:", flush=True)
    print(
        f"Validation {metric}: {summary['valid_metric']['mean']:.4f} ± {summary['valid_metric']['std']:.4f}",
        flush=True,
    )
    print(
        f"Test {metric}: {summary['test_metric']['mean']:.4f} ± {summary['test_metric']['std']:.4f}",
        flush=True,
    )

    return df


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run GINE experiments with or without virtual nodes"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of runs",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["ogbg-molpcba", "ogbg-molhiv"],
        default="ogbg-molpcba",
        help="Dataset name",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=400,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    parser.add_argument(
        "--virtual_node",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to run experiments with virtual nodes",
    )
    parser.add_argument(
        "--train_vn_eps",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to train virtual node epsilon",
    )
    parser.add_argument(
        "--vn_eps",
        type=float,
        default=0.0,
        help="Virtual node epsilon value",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    virtual_node = args.virtual_node.lower() == "true"

    # make results directory
    os.makedirs(f"{base_dir}/results", exist_ok=True)

    # Run experiments based on command line argument
    experiment_type = "with" if virtual_node else "without"
    print(
        f"Running experiments {experiment_type} Virtual Nodes for {args.dataset_name}",
        flush=True,
    )
    if args.output_file is None:
        output_file = f"{base_dir}/results/gine_results_{args.dataset_name}_{experiment_type}_virtual_node.csv"
    else:
        output_file = args.output_file

    results_df = run_multiple_experiments(
        dataset_name=args.dataset_name,
        n_runs=args.n_runs,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        virtual_node=virtual_node,
        train_vn_eps=args.train_vn_eps.lower() == "true",
        vn_eps=args.vn_eps,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        output_file=output_file,
    )

    print(results_df.to_string(), flush=True)
    print("Experiments completed successfully!", flush=True)
