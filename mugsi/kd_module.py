import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from kd_losses import (
    calc_KL_divergence, 
    calc_node_similarity, 
    nodeFeatureAlignment,
    calculate_conditional_probabilities,
    calculate_kl_loss
)

from ogb.graphproppred import Evaluator

class MuGSI_MLP_Student(pl.LightningModule):
    """
    For the ogbg-molpcba dataset (128 tasks).
    We'll do multi-label classification with BCEWithLogitsLoss, 
    plus the KD losses (soft label, node-sim, graph align, random-walk).
    """
    def __init__(
        self, 
        student_model: nn.Module,
        # Teacher knowledge
        teacher_ptr: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_h: torch.Tensor,
        teacher_g: torch.Tensor,
        # Hyperparams
        lr=1e-3,
        weight_decay=1e-5,
        # KD toggles
        use_soft_label=True,
        use_nodeSim=True,
        use_graphPooling=True,
        use_randomWalk=True,
        # KD coefficients
        softLabelReg=1.0,
        nodeSimReg=1.0,
        graphPoolingReg=1.0,
        randomWalkReg=1.0,
        sample_size=20,
        path_length=15,
        # Misc
        num_tasks=128,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["student_model","teacher_ptr","teacher_logits","teacher_h","teacher_g"])
        
        self.student_model = student_model
        self.teacher_ptr = teacher_ptr
        self.teacher_logits = teacher_logits
        self.teacher_h = teacher_h
        self.teacher_g = teacher_g
        
        # If dimension mismatch, add projection
        # self.proj_graph = None
        # if student_dim != teacher_dim:
        #     self.proj_graph = nn.Linear(student_dim, teacher_dim)
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        # toggles
        self.use_soft_label = use_soft_label
        self.use_nodeSim = use_nodeSim
        self.use_graphPooling = use_graphPooling
        self.use_randomWalk = use_randomWalk
        
        # KD regs
        self.softLabelReg = softLabelReg
        self.nodeSimReg = nodeSimReg
        self.graphPoolingReg = graphPoolingReg
        self.randomWalkReg = randomWalkReg
        
        self.sample_size = sample_size
        self.path_length = path_length
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator(name="ogbg-molpcba")

        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def forward(self, data):
        return self.student_model(data, output_emb=True)  # (logits, node_emb, graph_emb)
    
    def training_step(self, batch, batch_idx):
        logits_stu, node_emb_stu, graph_emb_stu = self(batch)
        y = batch.y  # shape [num_graphs_in_batch, 128]

        # =========== Ground Truth Loss ==============
        # We ignore "NaN" labels typical in molpcba; BCEWithLogits can handle them if masked carefully. 
        # But commonly in OGB, those "NaNs" are set as 0 with a mask. We'll do:
        is_labeled = ~torch.isnan(y)  # boolean mask
        gt_loss = self.criterion(
            logits_stu[is_labeled], 
            y[is_labeled].float()
        )
        
        # print(f"DEBUG: GT loss= {gt_loss}")

        # =========== KD Losses ===============
        kd_loss = torch.tensor(0.0, device=self.device)

        # A) Soft Label
        if self.use_soft_label:
            # teacher logits for the same graphs in the batch
            # We rely on batch.graph_id to pick them out
            tea_logits_batch = self.teacher_logits[batch.graph_id]
            kl_div = calc_KL_divergence(logits_stu, tea_logits_batch)
            kd_loss += self.softLabelReg * kl_div
            # print(f"DEBUG: soft label loss= {self.softLabelReg * kl_div}")

        # B) Node Similarity
        if self.use_nodeSim:
            # gather the teacher node embeddings 
            # using data.node_idx for each node in this batch
            node_idx = batch.node_idx
            print("===========================")
            print(f"DEBUG: node_idx shape = {node_idx.shape}")
            print(f"Node idx: {node_idx}")
            print("===========================")
            tea_node_emb_batch = self.teacher_h[node_idx]
            node_sim = calc_node_similarity(node_emb_stu, tea_node_emb_batch)
            kd_loss += self.nodeSimReg * node_sim
            # print(f"DEBUG: node sim loss= {self.nodeSimReg * node_sim}")

        # C) Graph-level alignment
        if self.use_graphPooling:
            tea_graph_emb_batch = self.teacher_g[batch.graph_id]
            graph_align_loss = nodeFeatureAlignment(graph_emb_stu, tea_graph_emb_batch)
            kd_loss += self.graphPoolingReg * graph_align_loss
            # print(f"DEBUG: Graph Pool loss= {self.graphPoolingReg * graph_align_loss}")
            
            
        if self.use_randomWalk:
            # data.random_walk_paths shape = [sample_size, path_length+1]
            random_walk_paths = batch.random_walk_paths
            node_idx = batch.node_idx
            tea_node_emb_batch = self.teacher_h[node_idx]

            if random_walk_paths.dim() == 2:
                teacher_cond = calculate_conditional_probabilities(random_walk_paths, tea_node_emb_batch)
                student_cond = calculate_conditional_probabilities(random_walk_paths, node_emb_stu)
                rw_kl = calculate_kl_loss(student_cond, teacher_cond)
                kd_loss += self.randomWalkReg * rw_kl


        # D) Random Walk Consistency
        # if self.use_randomWalk:
        #     random_walk_paths = batch.random_walk_paths  # shape: [B * sample_size, path_length+1]
        #     batch_idx = batch.batch  # shape: [total_nodes_in_batch], each node's graph ID
        #     num_graphs = batch.num_graphs
        #     s = self.sample_size
        #     node_idx = batch.node_idx  # shape [N_total_nodes_in_batch]
            
        #     # We'll accumulate random-walk loss for each graph, then average
        #     rw_loss_total = torch.tensor(0.0, device=self.device)
            
        #     for g_idx in range(num_graphs):
        #         # The random walks for the g_idx-th graph are in rows
        #         # [s*g_idx : s*(g_idx+1)] of random_walk_paths
        #         # => shape [sample_size, path_length+1]
        #         rw_chunk = random_walk_paths[s*g_idx : s*(g_idx+1)]

        #          # 2) Mask the node embeddings for only the nodes in g_idx
        #         mask_graph = (batch_idx == g_idx)
        #         teacher_node_emb_batch = self.teacher_h[node_idx]
        #         teacher_node_emb_graph = teacher_node_emb_batch[mask_graph]
        #         teacher_node_emb_graph = teacher_node_emb_batch[mask_graph]
        #         student_node_emb_graph = node_emb_stu[mask_graph]
        

        #         # Now compute teacher vs. student cond probabilities
        #         teacher_cond = calculate_conditional_probabilities(
        #             rw_chunk, teacher_node_emb_graph
        #         )
        #         student_cond = calculate_conditional_probabilities(
        #             rw_chunk, student_node_emb_graph
        #         )

        #         # KL over these distributions
        #         rw_kl_per_graph = calculate_kl_loss(student_cond, teacher_cond)
        #         rw_loss_total += rw_kl_per_graph
                
            
            # # E. Average over all graphs
            # rw_loss_total = rw_loss_total / num_graphs
            # kd_loss += self.randomWalkReg * rw_loss_total
            # print(f"DEBUG: Random Walk loss= {self.randomWalkReg * rw_loss_total}")
        
        total_loss = gt_loss + kd_loss
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            logits, _, _ = self.student_model(batch, output_emb=True)
            self.val_preds.append(logits.cpu())
            self.val_targets.append(batch.y.cpu())

    def on_validation_epoch_end(self):
        y_pred = torch.cat(self.val_preds, dim=0).numpy()
        y_true = torch.cat(self.val_targets, dim=0).numpy()
        
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        result = self.evaluator.eval(input_dict)
        ap_val = result["ap"]
        self.log("valid_ap", ap_val, prog_bar=True)
        
        self.val_preds = []
        self.val_targets = []
        print(f"[Val] AP = {ap_val:.4f}")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            logits, _, _ = self.student_model(batch, output_emb=True)
            self.test_preds.append(logits.cpu())
            self.test_targets.append(batch.y.cpu())

    def on_test_epoch_end(self):
        y_pred = torch.cat(self.test_preds, dim=0).numpy()
        y_true = torch.cat(self.test_targets, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        result = self.evaluator.eval(input_dict)
        ap_test = result["ap"]
        self.log("test_ap", ap_test, prog_bar=True)
        print(f"[Test] AP = {ap_test:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
