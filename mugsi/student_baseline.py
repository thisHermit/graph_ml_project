import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from molpcba_transforms import MolPCBA_Transform

import sys
sys.path.append('../')

from model.MLP import MLP, GA_MLP

class MolPcbaModule(pl.LightningModule):
    """
    For multi-task classification on ogbg-molpcba with MLP or GA_MLP baseline.
    We use BCEWithLogitsLoss with a mask for missing labels.
    """
    def __init__(self, 
                 model_name='MLP', 
                 node_dim=9, 
                 hidden_dim=200, 
                 edge_dim=3, 
                 lr=1e-3,
                 weight_decay=1e-6,
                 pooling_method='sum',
                 use_lappe=False,
                 use_louvain=False,
                 use_khop=False):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim  # if GA_MLP used
        self.num_tasks = 128
        self.pooling_method = pooling_method
        self.lr = lr
        self.weight_decay = weight_decay

        # Build student model
        if self.model_name == 'MLP':
            self.student_model = MLP(node_dim=self.node_dim,
                                     hidden_dim=self.hidden_dim,
                                     num_classes=self.num_tasks,
                                     pooling_method=self.pooling_method)
        else:
            # GA_MLP
            self.student_model = GA_MLP(node_dim=self.node_dim,
                                        edge_dim=self.edge_dim,
                                        hidden_dim=self.hidden_dim,
                                        num_classes=self.num_tasks,
                                        pooling_method=self.pooling_method)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator(name="ogbg-molpcba")

    def forward(self, data):
        return self.student_model(data)

    def training_step(self, batch, batch_idx):
        y = batch.y  # shape [batch_size, 128]
        # mask missing labels
        mask = ~torch.isnan(y)
        pred = self(batch)  # [batch_size, 128]
        # BCE only on labeled entries
        loss = self.loss_fn(pred[mask], y[mask].float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        y = batch.y
        mask = ~torch.isnan(y)
        pred = self(batch)
        # compute AP
        input_dict = {'y_true': y.cpu().numpy(), 'y_pred': pred.cpu().numpy()}
        result_dict = self.evaluator.eval(input_dict)
        ap = result_dict["ap"]
        self.log("val_ap", ap, prog_bar=True)
        return ap

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        y = batch.y
        pred = self(batch)
        input_dict = {'y_true': y.cpu().numpy(), 'y_pred': pred.cpu().numpy()}
        result_dict = self.evaluator.eval(input_dict)
        ap = result_dict["ap"]
        self.log("test_ap", ap, prog_bar=True)
        return ap

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    
def main():
    parser = argparse.ArgumentParser(description="Train a baseline MLP or GA-MLP on ogbg-molpcba.")
    # Model and training parameters
    parser.add_argument("--model_name", type=str, default="MLP", choices=["MLP","GA_MLP"],
                        help="Choose between 'MLP' and 'GA_MLP' as the student model.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension size for the student model.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay (L2 regularization) for the optimizer.")
    parser.add_argument("--max_epochs", type=int, default=30,
                        help="Maximum number of training epochs.")
    parser.add_argument("--pooling_method", type=str, default="sum", choices=["sum","attention"],
                        help="Pooling method to aggregate node features into graph features.")
    # Device and data loading parameters
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID to use. Defaults to 0.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker threads for data loading.")
    # Transform options
    parser.add_argument("--use_lappe", action='store_true', help="Enable Laplacian Positional Encoding (LapPE) in transform.")
    parser.add_argument("--use_louvain", action='store_true', help="Enable Louvain clustering in transform.")
    parser.add_argument("--use_khop", action='store_true', help="Enable KHop transform for GA-MLP.")
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    
    # Determine whether to use KHop based on model choice or explicit flag
    use_khop = args.use_khop or (args.model_name == "GA_MLP")
    
    # Initialize the transform pipeline
    transform = MolPCBA_Transform(
        use_khop=use_khop, 
        use_louvain=args.use_louvain, 
        use_lappe=args.use_lappe,
        khop_list=[1]  # 1-hop for GA_MLP
    )
    
    if args.use_lappe and args.use_louvain:
        cache_dir = './dataset/ogbg-molpcba_lappe_louvain'
    elif args.use_lappe and args.use_khop:
        cache_dir = './dataset/ogbg-molpcba_lappe_khop'
    elif args.use_lappe:
        cache_dir = './dataset/ogbg-molpcba_lappe'
    else:
        cache_dir = './dataset/ogbg-molpcba_raw'

    # Load dataset
    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", transform=transform, root=cache_dir)
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    val_dataset   = dataset[split_idx['valid']]
    test_dataset  = dataset[split_idx['test']]
    
    # Debug: Inspect a sample graph
    print("==="*20)
    print("DEBGUG")
    sample_graph = train_dataset[0]
    print("Sample Graph Data Types:")
    print(f"x: {sample_graph.x.dtype}, shape: {sample_graph.x.shape}")  # Should be torch.float32
    print(f"hop1_feature: {sample_graph.hop1_feature.dtype} shape: {sample_graph.hop1_feature.shape}" if hasattr(sample_graph, 'hop1_feature') else "No hop1_feature")
    print(f"edge_features: {sample_graph.edge_features.dtype}, shape: {sample_graph.edge_features.shape}" if hasattr(sample_graph, 'edge_features') else "No edge_features")
    print(f"y: {sample_graph.y.dtype}  shape: {sample_graph.y.shape}")  # Should be torch.float32

    print("==="*20)

     # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # node_dim from dataset.num_features
    node_dim = dataset.num_features # 9 + LapPE
    
    edge_dim = 3 if args.model_name=="GA_MLP" else 0   # ogbg-molpcba has edge_attr dimension=3
    
    print(f"Dataset num_features: {dataset.num_features}")
    print(f"use_lappe: {args.use_lappe}")
    print(f"Calculated node_dim: {node_dim}")
    print(f"Edge dimensions: {edge_dim}")

    model_module = MolPcbaModule(model_name=args.model_name,
                                 node_dim=node_dim,
                                 hidden_dim=args.hidden_dim,
                                 edge_dim=edge_dim,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 pooling_method=args.pooling_method,
                                 use_lappe=args.use_lappe,
                                 use_louvain=args.use_louvain,
                                 use_khop=use_khop)

    # 2) Trainer
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=[args.device] if torch.cuda.is_available() else 1,
                         log_every_n_steps=100)
    # 3) Fit
    trainer.fit(model_module, train_loader, val_loader)

    # 4) Test
    test_result = trainer.test(model_module, test_loader)
    print("Final test result:", test_result)

if __name__ == "__main__":
    main()