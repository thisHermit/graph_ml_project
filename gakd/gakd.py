import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import json
import time
from tqdm import tqdm
import argparse
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from baseline import GINENetwork
import pandas as pd
from datetime import datetime
import torch_geometric.nn as nng

base_dir = os.getenv(
    "BASE_DIR",
    f"/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GakD",
)
if not os.path.exists(base_dir):
    os.makedirs(base_dir, exist_ok=True)


class logits_D(nn.Module):
    """
    Logits Identifier (Section 3.3)
    Purpose: Discriminate between student and teacher logits
    Description: A simple MLP with residual learning and hidden layer with dimension equal to the number of classes.
    Input: Model logits (N, C)
    Output: Predicted class label for the logits (N, C+1)

    The output layer has one more dimension than the number of classes to account for the probability of being a student or teacher.
    The output predicts class label for the logits as it appears to be more stable according to the authors.
    """

    def __init__(self, n_class, n_hidden):
        super(logits_D, self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.lin = nn.Linear(self.n_class, self.n_hidden)  # assert n_class==n_hidden
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.n_hidden, self.n_class + 1, bias=False)

    def forward(self, logits, temperature=1.0):
        out = self.lin(logits / temperature)
        out = logits + out
        out = self.relu(out)
        dist = self.lin2(out)
        return dist


class local_emb_D(nn.Module):
    """
    Representation Identifier (Embeddings): D_e_Local Embedding Discriminator (Section 3.2)
    Purpose: Learns a weight matrix that maps the embedding representations of two nodes from the same graph and
             same model to a real value encoding the affinity between the two.
             It encourages the student to inherit the local affinity hidden in teacher's node embeddings
    Description: A parameteric weight matrix that is learned during training.
    Input: Model embeddings - (N, D)
    Output: Affinity score - (N, N)

    N: Number of nodes in the graph
    D: Dimension of the embeddings
    """

    def __init__(self, n_hidden):
        super(local_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden,)))
        self.scale = nn.Parameter(torch.full(size=(1,), fill_value=1.0))

    def forward(self, emb, batch):
        emb = F.normalize(emb, p=2)
        (u, v) = batch.edge_index
        euw = emb[u] @ torch.diag(self.d)
        pair_dis = euw @ emb[v].t()
        return torch.diag(pair_dis) * self.scale


class global_emb_D(nn.Module):
    """
    Representation Identifier (Summary): D_e_Global Embedding Discriminator (Section 3.2)
    Purpose: Learns a weight matrix that maps the embedding representations of nodes from a model to a
             summary embedding representation (Mean of the graph'snode embeddings) from a model.
             Both models could be same or different.
             It encourages the student to inherit the global affinity
    Description: A parameteric weight matrix that is learned during training.
    Input:
     - Model embeddings - (N, D)
     - Summary embeddings - (1, D)
    Output: Affinity score - (N, 1)

    N: Number of nodes in the graph
    D: Dimension of the embeddings
    """

    def __init__(self, n_hidden):
        super(global_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden,)))
        self.scale = nn.Parameter(torch.full(size=(1,), fill_value=1.0))

    def forward(self, emb, summary, batch):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d)
        sims = []
        for i, s in enumerate(summary):
            pre, post = batch.ptr[i], batch.ptr[i + 1]
            sims.append(sim[pre:post] @ s.unsqueeze(-1))
        sim = torch.cat(sims, dim=0).squeeze(-1)
        return sim * self.scale


def load_knowledge(kd_path, device):  # load teacher knowledge
    """
    Load teacher knowledge from a file.
    Input:
     - kd_path: Path to the teacher knowledge file
     - device: Device to load the knowledge on
    Output:
     - tea_logits: Teacher logits (N, C)
     - tea_h: Teacher embeddings (N, D)
     - tea_g: Teacher summary embeddings (1, D)
     - new_ptr: Teacher pointer (N) - cumulative sum of the number of nodes in each graph,
                                      used to index the nodes in the graph
    """
    assert os.path.isfile(kd_path), "Please download teacher knowledge first"
    knowledge = torch.load(kd_path, map_location=device)
    tea_logits = knowledge["logits"].float()
    tea_h = knowledge["h-embedding"]
    tea_g = knowledge["g-embedding"]
    new_ptr = knowledge["ptr"]
    return tea_logits, tea_h, tea_g, new_ptr


class AddGraphIdTransform:
    """
    Add a graph id to the OGB dataset during loading.
    """

    def __init__(self):
        self.graph_id = 0

    def __call__(self, data):
        data.graph_id = self.graph_id
        self.graph_id += 1
        return data


class GAKD_trainer:
    """
    GAKD Trainer

    This class is responsible for training the student model under the GAKD framework.

    The class is initialized with the following parameters:
    - student_model_args: Dictionary containing the student model's architecture and training parameters.
    - teacher_knowledge_path: Path to the teacher knowledge file.
    - dataset_name: Name of the dataset to use for training.
    - student_optimizer_lr: Learning rate for the student optimizer.
    - student_optimizer_weight_decay: Weight decay for the student optimizer.
    - discriminator_optimizer_lr: Learning rate for the discriminator optimizer.
    - discriminator_optimizer_weight_decay: Weight decay for the discriminator optimizer.
    - batch_size: Batch size for training.
    - num_workers: Number of workers for data loading.
    - discriminator_update_freq: Frequency of discriminator updates. (K in paper, Section C.1: Experimental Details)
    - train_discriminator_logits: Whether to train the logits identifier.
    - train_discriminator_embeddings: Whether to train the embeddings identifier.
    - epochs: Number of epochs to train for.
    - seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        student_model_args: dict,
        teacher_knowledge_path: str,
        dataset_name="ogbg-molpcba",
        student_optimizer_lr=5e-3,
        student_optimizer_weight_decay=1e-5,
        discriminator_optimizer_lr=1e-2,
        discriminator_optimizer_weight_decay=5e-4,
        batch_size=32,
        num_workers=4,
        discriminator_update_freq=5,  # K in paper
        train_discriminator_logits=True,
        train_discriminator_embeddings=True,
        epochs=100,
        seed=42,
    ):
        self.seed = seed
        self.dataset_name = dataset_name
        self.student_model_args = student_model_args
        self.teacher_knowledge_path = teacher_knowledge_path
        self.embedding_dim = student_model_args["embedding_dim"]
        self.student_lr = student_optimizer_lr
        self.student_weight_decay = student_optimizer_weight_decay
        self.discriminator_lr = discriminator_optimizer_lr
        self.discriminator_weight_decay = discriminator_optimizer_weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.discriminator_update_freq = discriminator_update_freq
        self.train_discriminator_logits = train_discriminator_logits
        self.train_discriminator_embeddings = train_discriminator_embeddings
        self.setup()

    def setup(self):
        self._set_device()
        self._set_seed(self.seed)
        self._load_dataset()
        self._load_knowledge()
        self._configure_student_model()
        self._setup_student_optimizer()
        self._setup_discriminator()
        self.evaluate_teacher()

    def _configure_student_model(self):
        """
        Configure the student model.
        This function sets the embedding dimension and output dimension of the student model.
        - It also checks for embedding dimension mismatch between teacher and student.
        - If the embedding dimension is not specified in the student model arguments, it is set to the teacher's embedding dimension.
        - If the output dimension is not specified in the student model arguments, it is set to the number of tasks in the dataset.
        """
        if self.student_model_args is not None:
            if self.student_model_args["embedding_dim"] is None:
                self.student_model_args["embedding_dim"] = self._teacher_h_dim
                self.embedding_dim = self._teacher_h_dim
            else:
                assert (
                    self.student_model_args["embedding_dim"] == self._teacher_h_dim
                ), "Embedding dimension mismatch between teacher and student"
            if self.student_model_args["out_dim"] is None:
                self.student_model_args["out_dim"] = self.dataset.num_tasks

        self.student_model = GINENetwork(
            hidden_dim=self.student_model_args["embedding_dim"],
            out_dim=self.student_model_args["out_dim"],
            num_layers=self.student_model_args["num_layers"],
            dropout=self.student_model_args["dropout"],
            virtual_node=self.student_model_args["virtual_node"],
            train_vn_eps=self.student_model_args["train_vn_eps"],
            vn_eps=self.student_model_args["vn_eps"],
            return_embeddings=True,
        )

    def _load_dataset(self):
        """
        Load the dataset.
        - It creates the directory for storing the dataset if it doesn't exist.
        - It adds a graph id to the dataset during loading.
        - It loads the dataset and splits it into train, validation, and test sets.
        - It initializes the dataloaders for training, validation, and testing.
        - It initializes the OGB dataset evaluator for evaluating the model's performance.
        """
        os.makedirs(f"{base_dir}/data", exist_ok=True)
        self._transform = AddGraphIdTransform()
        self.dataset = PygGraphPropPredDataset(
            name=self.dataset_name,
            root=f"{base_dir}/data",
            pre_transform=self._transform,
        )
        self.split_idx = self.dataset.get_idx_split()

        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.dataset[self.split_idx["train"]],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.valid_loader = DataLoader(
            self.dataset[self.split_idx["valid"]],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            self.dataset[self.split_idx["test"]],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.evaluator = Evaluator(name=self.dataset_name)

    def _setup_student_optimizer(self):
        """
        Setup the student optimizer.
        """
        self.student_model = self.student_model.to(self.device)
        self.student_optimizer = optim.Adam(
            self.student_model.parameters(),
            lr=self.student_lr,
            weight_decay=self.student_weight_decay,
        )

    def _setup_discriminator(self):
        """
        Setup the discriminators.
        - It initializes the local and global embedding discriminators.
        - It initializes the logits discriminator.
        - It sets up the optimizer and label loss criterion for the discriminators.
        """
        self.discriminator_e_local = local_emb_D(n_hidden=self.embedding_dim).to(
            self.device
        )
        self.discriminator_e_global = global_emb_D(n_hidden=self.embedding_dim).to(
            self.device
        )
        self.discriminator_logits = logits_D(
            n_class=self.dataset.num_tasks, n_hidden=self.dataset.num_tasks
        ).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            [
                {
                    "params": self.discriminator_e_local.parameters(),
                },
                {
                    "params": self.discriminator_e_global.parameters(),
                },
                {
                    "params": self.discriminator_logits.parameters(),
                },
            ],
            lr=self.discriminator_lr,
            weight_decay=self.discriminator_weight_decay,
        )
        self.discriminator_loss = torch.nn.BCELoss()
        self.class_criterion = torch.nn.BCEWithLogitsLoss()
        self._train_ids = self.split_idx["train"].to(self.device)

    def _set_device(self):
        """
        Set the device for training.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS is currently slower than CPU due to missing int64 min/max ops
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}", flush=True)
        self.device = device

    def _load_knowledge(self):
        """
        Load teacher knowledge.
        - It checks if the teacher knowledge file exists.
        - It loads the teacher knowledge and selects the graph indices for the training set.
        - It initializes the teacher embeddings and logits.
        """
        print(self.teacher_knowledge_path, os.path.isfile(self.teacher_knowledge_path))
        assert os.path.isfile(
            self.teacher_knowledge_path
        ), "Please download teacher knowledge first"
        knowledge = torch.load(self.teacher_knowledge_path, map_location=self.device)

        # Load teacher logits
        self.teacher_logits = knowledge["logits"].float().to(self.device)
        self.teacher_logits = self.teacher_logits[self.split_idx["train"]]
        print("Teacher logits Dimension: ", self.teacher_logits.shape, flush=True)
        self._teacher_logits_dim = self.teacher_logits.shape[1]

        # Load teacher summary embeddings
        self.teacher_g = knowledge["g-embeddings"].to(self.device)
        self.teacher_g = self.teacher_g[self.split_idx["train"]]
        print("Teacher g (summary) dimension: ", self.teacher_g.shape, flush=True)
        self._teacher_g_dim = self.teacher_g.shape[1]

        # Load teacher ptr: cumulative sum of the number of nodes in each graph,
        # used to index the nodes in the graph
        self.teacher_ptr = knowledge["ptr"].to(self.device)

        # Load teacher embeddings
        self.teacher_h = knowledge["h-embeddings"].to(self.device)
        pre, post = self.teacher_ptr[:-1], self.teacher_ptr[1:]
        train_pre, train_post = (
            pre[self.split_idx["train"]],
            post[self.split_idx["train"]],
        )
        # Determine the indices of the nodes in the training set
        self.teacher_h_idx = torch.cat(
            [
                torch.arange(pre, post)
                for pre, post in list(zip(*[train_pre, train_post]))
            ],
            dim=0,
        )
        self.teacher_h = self.teacher_h[self.teacher_h_idx]
        self._teacher_h_dim = self.teacher_h.shape[1]
        print("Teacher h (embedding) dimension: ", self.teacher_h.shape, flush=True)

        # Calculate the number of nodes in each graph included in the training set
        nodes_count = torch.tensor(
            [(post - pre).item() for pre, post in list(zip(train_pre, train_post))]
        )
        # Update the teacher ptr to reflect the cumulative sum of the number of nodes in each graph in the training set
        self.teacher_ptr = torch.cat(
            [torch.tensor([0]), torch.cumsum(nodes_count, dim=0)]
        ).to(self.device)
        print("Teacher ptr shape: ", self.teacher_ptr.shape, flush=True)

    def evaluate_teacher(self):
        """
        Evaluate the teacher logits on the training set using the OGB evaluator.
        """
        train_y_true = self.dataset[self.split_idx["train"]].y
        train_y_pred = self.teacher_logits
        input_dict = {"y_true": train_y_true, "y_pred": train_y_pred}
        if self.dataset_name == "ogbg-molpcba":
            print(
                f"Teacher performance on Training set: {self.evaluator.eval(input_dict)['ap']}",
                flush=True,
            )
        else:
            print(
                f"Teacher performance on Training set: {self.evaluator.eval(input_dict)['rocauc']}",
                flush=True,
            )

    def _set_seed(self, seed=42):
        """
        Set the seed for reproducibility.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}", flush=True)

    def _get_batch_idx_from_teacher(self, batch):
        """
        Map the batch graph indices to the teacher knowledge graph's indices.
        - It first maps graph ids of each node in batch to the training set graph's id indexes.
        - It then extracts the node indices of each graph included in the batch coming from training set
          using the teacher ptr.

        Returns:
            batch_graph_idx: Tensor of shape (batch_size,), containing the graph indices of the batch preserving order
            batch_node_idx: Tensor of shape (batch_size,), containing the node indices of the batch.
        """
        new_pre = self.teacher_ptr[:-1]
        new_post = self.teacher_ptr[1:]
        # Map graph ids of each node in batch to training set graph's indices
        # For e,g
        # new_pre, new_post = [0, 3, 6,.. 12], [3, 6, 9,.. 15], len(new_pre) = len(new_post) = 10 + 1 (10 is the number of graphs in training set)
        # batch.batch = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] <- zero-indexed graph ids for each node in batch
        # assumed true graph ids are [4, 4, 7, 7, 10, 10, 13, 13, 16, 16]
        # self._train_ids = [2, 4, 5, 7, 8, 10, 11, 13, 14, 16] <- zero-indexed graph ids for each node in training set
        # batch_graph_idx = [1, 3, 5, 7, 9] <- indices of the graphs included in the batch,
        #                                      used to extract the node indices of each graph included in the batch from batch_pre, batch_post
        # batch_pre, batch_post = selected from new_pre, new_post for only the graphs included in the batch using batch_graph_idx
        new_ids = [
            (self._train_ids == batch.graph_id[vid]).nonzero().item()
            for vid in batch.batch
        ]
        # batch_graph_idx = torch.tensor(new_ids, device=self.device)
        batch_graph_idx = torch.tensor(new_ids, device=self.device).unique_consecutive()
        # Extract the pre and post indices of the graphs included in the batch
        batch_pre = new_pre[batch_graph_idx]
        batch_post = new_post[batch_graph_idx]
        # Extract the node indices of each graph included in the batch
        batch_node_idx = torch.cat(
            [
                torch.arange(pre, post)
                for pre, post in list(zip(*[batch_pre, batch_post]))
            ],
            dim=0,
        ).to(self.device)
        # Return the unique consecutive graph indices and the node indices of each graph included in the batch
        return batch_graph_idx.to(self.device), batch_node_idx.to(self.device)

    def _train_batch(self, batch, epoch):
        """
        Train a single batch of data under the GAKD framework.
        """
        batch = batch.to(self.device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            return 0

        student_batch_pred, student_batch_h = self.student_model(
            batch
        )  # [batch_size, 1], [N, h_dim] where N is the number of nodes in the batch
        student_batch_g = nng.global_mean_pool(
            student_batch_h, batch.batch
        )  # [G, h_dim] where G is the number of graphs in the batch
        self.student_optimizer.zero_grad()
        y_true = batch.y.float()
        # Filter out NAN class labels
        y_labeled = batch.y == batch.y

        # classification loss, to be added to student loss for logits identifier in training student
        class_loss = self.class_criterion(
            student_batch_pred.float()[y_labeled], y_true[y_labeled]
        )
        # Get the batch indices from the teacher knowledge
        # This is used to extract the teacher embeddings and logits for the batch
        # The teacher embeddings and logits are used to train the discriminator
        batch_graph_idx, batch_node_idx = self._get_batch_idx_from_teacher(batch)

        teacher_batch_h = self.teacher_h[batch_node_idx].to(self.device)
        teacher_batch_g = self.teacher_g[batch_graph_idx].to(self.device)
        teacher_batch_logits = self.teacher_logits[batch_graph_idx].to(self.device)

        assert (
            teacher_batch_h.shape == student_batch_h.shape
        ), "Teacher and student batch h shapes do not match"
        assert (
            teacher_batch_g.shape == student_batch_g.shape
        ), "Teacher and student batch g shapes do not match"
        assert (
            teacher_batch_logits.shape == student_batch_pred.shape
        ), "Teacher and student batch logits shapes do not match"

        #### Train discriminator, only update discriminator every self.discriminator_update_freq epochs
        if epoch % self.discriminator_update_freq == 0:
            discriminator_loss = 0

            ## train logits identifier: D_l
            if self.train_discriminator_logits:
                # Section 3.3: Implementation of equation 3 (logits identifier)
                self.discriminator_logits.train()
                # detach student logits to avoid backprop through student for discriminator training
                student_logits = student_batch_pred.detach()
                # D_l(z_teacher)
                z_teacher = self.discriminator_logits(teacher_batch_logits)
                # D_l(z_student)
                z_student = self.discriminator_logits(student_logits)
                # Real | D_l(z_teacher)
                prob_real_given_z = torch.sigmoid(z_teacher[:, -1])
                # Fake | D_l(z_student)
                prob_fake_given_z = torch.sigmoid(z_student[:, -1])
                # logP(Real | D_l(z_teacher)) + logP(Fake | D_l(z_student)) - First half of equation 3
                adversarial_logits_loss = self.discriminator_loss(
                    prob_real_given_z, torch.ones_like(prob_real_given_z)
                ) + self.discriminator_loss(
                    prob_fake_given_z, torch.zeros_like(prob_fake_given_z)
                )
                # logP(y_v | D_l(z_teacher)) where y_v is the true class labels for node v
                y_v_given_z_pos = self.class_criterion(
                    z_teacher[:, :-1][y_labeled], y_true[y_labeled]
                )
                # logP(y_v | D_l(z_student))
                y_v_given_z_neg = self.class_criterion(
                    z_student[:, :-1][y_labeled], y_true[y_labeled]
                )
                # logP(y_v | D_l(z_teacher)) + logP(y_v | D_l(z_student)) - Second half of equation 3
                label_loss = y_v_given_z_pos + y_v_given_z_neg
                # Add the adversarial logits loss and label loss to the discriminator loss - equation 3
                discriminator_loss += 0.5 * (adversarial_logits_loss + label_loss)

            ## train local embedding representation identifier: D_e_local
            if self.train_discriminator_embeddings:
                # Section 3.2: Implementation of equation 1 (local embedding identifier) - Maximization for discriminator

                # Set the local embedding discriminator to train mode
                self.discriminator_e_local.train()
                # D_e_local(teacher_embeddings)
                pos_e = self.discriminator_e_local(teacher_batch_h, batch)
                # D_e_local(student_embeddings)
                neg_e = self.discriminator_e_local(student_batch_h.detach(), batch)
                # Real | D_e_local(teacher_embeddings)
                prob_real_given_e = torch.sigmoid(pos_e)
                # Fake | D_e_local(student_embeddings)
                prob_fake_given_e = torch.sigmoid(neg_e)
                # J_local = logP(Real | D_e_local(teacher_embeddings)) + logP(Fake | D_e_local(student_embeddings)) - First half of equation 1
                adverserial_local_e_loss = self.discriminator_loss(
                    prob_real_given_e, torch.ones_like(prob_real_given_e)
                ) + self.discriminator_loss(
                    prob_fake_given_e, torch.zeros_like(prob_fake_given_e)
                )

                ## train global embedding representation identifier: D_e_global
                # Set the global embedding discriminator to train mode
                self.discriminator_e_global.train()
                # Summary teacher (S_teacher)
                teacher_summary = torch.sigmoid(teacher_batch_g)
                # D_e_global(teacher_embeddings, S_teacher)
                e_teacher_summary_teacher = self.discriminator_e_global(
                    teacher_batch_h, teacher_summary, batch
                )
                # D_e_global(student_embeddings, S_teacher)
                e_student_summary_teacher = self.discriminator_e_global(
                    student_batch_h.detach(), teacher_summary, batch
                )
                # Real | D_e_global(teacher_embeddings, S_teacher)
                prob_real_given_e_global = torch.sigmoid(e_teacher_summary_teacher)
                # Fake | D_e_global(student_embeddings, S_teacher)
                prob_fake_given_e_global = torch.sigmoid(e_student_summary_teacher)
                # logP(Real | D_e_global(teacher_embeddings, S_teacher)) + logP(Fake | D_e_global(student_embeddings, S_teacher))
                adverserial_global_e_loss1 = self.discriminator_loss(
                    prob_real_given_e_global,
                    torch.ones_like(prob_real_given_e_global),
                ) + self.discriminator_loss(
                    prob_fake_given_e_global,
                    torch.zeros_like(prob_fake_given_e_global),
                )
                # Summary student (S_student)
                student_summary = torch.sigmoid(student_batch_g)
                # D_e_global(student_embeddings, S_student)
                e_student_summary_student = self.discriminator_e_global(
                    student_batch_h.detach(), student_summary.detach(), batch
                )
                # D_e_global(teacher_embeddings, S_student)
                e_teacher_summary_student = self.discriminator_e_global(
                    teacher_batch_h, student_summary.detach(), batch
                )
                # logP(Real | D_e_global(student_embeddings, S_student))
                prob_real_given_e_global = torch.sigmoid(e_student_summary_student)
                # logP(Fake | D_e_global(teacher_embeddings, S_student))
                prob_fake_given_e_global = torch.sigmoid(e_teacher_summary_student)
                # logP(Real | D_e_global(student_embeddings, S_student)) + logP(Fake | D_e_global(teacher_embeddings, S_student))
                adverserial_global_e_loss2 = self.discriminator_loss(
                    prob_real_given_e_global,
                    torch.ones_like(prob_real_given_e_global),
                ) + self.discriminator_loss(
                    prob_fake_given_e_global,
                    torch.zeros_like(prob_fake_given_e_global),
                )
                # Equation 1
                discriminator_loss = (
                    discriminator_loss
                    # J_local = logP(Real | D_e_local(teacher_embeddings)) + logP(Fake | D_e_local(student_embeddings))
                    + adverserial_local_e_loss
                    # J_global = logP(Real | D_e_global(teacher_embeddings, S_teacher)) + logP(Fake | D_e_global(student_embeddings, S_teacher))  +
                    #            logP(Real | D_e_global(student_embeddings, S_student)) + logP(Fake | D_e_global(teacher_embeddings, S_student))
                    + adverserial_global_e_loss1
                    + adverserial_global_e_loss2
                )
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        #### Train student
        student_loss = class_loss

        ## fooling logits discriminator
        if self.train_discriminator_logits:
            self.discriminator_logits.eval()
            # Section 3.3: Implementation of equation 4 (logits identifier) - Minimization for student
            # only keeping the student terms
            z_teacher = self.discriminator_logits(teacher_batch_logits)
            z_student = self.discriminator_logits(student_batch_pred)
            # Fake | D_l(z_student)
            prob_fake_given_z = torch.sigmoid(z_student[:, -1])
            # logP(Fake | D_l(z_student)) - equation 4
            adversarial_logits_loss = self.discriminator_loss(
                prob_fake_given_z, torch.ones_like(prob_fake_given_z)
            )
            # logP(y_v | D_l(z_student))
            label_loss = self.class_criterion(
                z_student[:, :-1][y_labeled], y_true[y_labeled]
            )
            # L1 loss
            l1_loss = (
                torch.norm(student_batch_pred - teacher_batch_logits, p=1)
                * 1
                / len(batch.batch)
            )
            # Equation 4
            student_loss = (
                student_loss + 0.5 * (adversarial_logits_loss + label_loss) + l1_loss
            )

        ## fooling local embedding representation identifier
        if self.train_discriminator_embeddings:
            self.discriminator_e_local.eval()
            # only keeping the student terms
            neg_e = self.discriminator_e_local(student_batch_h, batch)
            # Fake | D_e_local(student_embeddings)
            prob_fake_given_e = torch.sigmoid(neg_e)
            # logP(Fake | D_e_local(student_embeddings))
            adversarial_local_e_loss = self.discriminator_loss(
                prob_fake_given_e, torch.ones_like(prob_fake_given_e)
            )

            ## fooling global embedding representation identifier
            self.discriminator_e_global.eval()
            # Summary teacher (S_teacher)
            teacher_summary = torch.sigmoid(teacher_batch_g)
            # D_e_global(student_embeddings, S_teacher)
            e_student_summary_teacher = self.discriminator_e_global(
                student_batch_h, teacher_summary, batch
            )
            # Fake | D_e_global(student_embeddings, S_teacher)
            prob_fake_given_e_global = torch.sigmoid(e_student_summary_teacher)
            # logP(Fake | D_e_global(student_embeddings, S_teacher))
            adverserial_global_e_loss1 = self.discriminator_loss(
                prob_fake_given_e_global, torch.ones_like(prob_fake_given_e_global)
            )
            # Summary student (S_student)
            student_summary = torch.sigmoid(student_batch_g)
            # D_e_global(teacher_embeddings, S_student)
            e_teacher_summary_student = self.discriminator_e_global(
                teacher_batch_h, student_summary, batch
            )
            # D_e_global(student_embeddings, S_student)
            e_student_summary_student = self.discriminator_e_global(
                student_batch_h, student_summary, batch
            )
            # Real | D_e_global(student_embeddings, S_student)
            prob_real_given_e_global = torch.sigmoid(e_student_summary_student)
            # Fake | D_e_global(teacher_embeddings, S_student)
            prob_fake_given_e_global = torch.sigmoid(e_teacher_summary_student)
            # logP(Real | D_e_global(student_embeddings, S_student)) + logP(Fake | D_e_global(teacher_embeddings, S_student))
            adverserial_global_e_loss2 = self.discriminator_loss(
                prob_real_given_e_global, torch.zeros_like(prob_real_given_e_global)
            ) + self.discriminator_loss(
                prob_fake_given_e_global, torch.ones_like(prob_fake_given_e_global)
            )
            student_loss = (
                student_loss
                + adversarial_local_e_loss
                + adverserial_global_e_loss1
                + adverserial_global_e_loss2
            )

        self.student_optimizer.zero_grad()
        student_loss.backward()
        self.student_optimizer.step()

        return student_loss.item()

    def train(self):
        """
        Train the student model under the GAKD framework for a given number of epochs.
        """
        best_valid_ap = 0
        for epoch in range(self.epochs):
            print("Epoch: ", epoch + 1, flush=True)
            self.student_model.train()
            train_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx % 100 == 0:
                    print(
                        "Processing training batch {}/{}, size: {}".format(
                            batch_idx + 1, len(self.train_loader), batch.y.shape[0]
                        ),
                        flush=True,
                    )
                batch_loss = self._train_batch(batch, epoch)
                train_loss += batch_loss

            train_loss /= len(self.train_loader)

            if epoch % 5 == 0:
                valid_ap = self.evaluate(split="valid")
                print(
                    f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid AP: {valid_ap:.4f}",
                    flush=True,
                )

                # Save best model
                if valid_ap > best_valid_ap:
                    best_valid_ap = valid_ap
                    os.makedirs(f"{base_dir}/models", exist_ok=True)
                    torch.save(
                        self.student_model.state_dict(),
                        f"{base_dir}/models/gine_student_kd_{self.dataset_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pt",
                    )

    def evaluate(self, split="valid"):
        """
        Evaluate the student model on the validation or test set.
        """
        self.student_model.eval()
        loader = self.valid_loader if split == "valid" else self.test_loader
        y_true_list = []
        y_pred_list = []

        for batch in loader:
            batch = batch.to(self.device)
            if batch.x.shape[0] == 1:
                continue
            with torch.no_grad():
                y_pred, _ = self.student_model(batch)
            y_true_list.append(batch.y.view(y_pred.shape).detach().cpu())
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
    teacher_knowledge_path,
    dataset_name="ogbg-molpcba",
    n_runs=5,
    include_vn_student=True,
    student_optimizer_lr=5e-3,
    student_optimizer_weight_decay=1e-5,
    discriminator_optimizer_lr=1e-2,
    discriminator_optimizer_weight_decay=5e-4,
    batch_size=32,
    num_workers=4,
    discriminator_update_freq=5,
    epochs=100,
    train_discriminator_logits=True,
    train_discriminator_embeddings=True,
    output_file=f"{base_dir}/results/gine_student_gakd_molpcba.csv",
    student_model_args=None,
):
    """
    Run multiple experiments with default or user-defined configurations.
    """
    results = []
    metric = "ap" if dataset_name == "ogbg-molpcba" else "rocauc"
    for run in range(n_runs):
        print(f"\nStarting Run {run + 1}/{n_runs}", flush=True)
        seed = 42 + run
        trainer = GAKD_trainer(
            student_model_args=student_model_args,
            teacher_knowledge_path=teacher_knowledge_path,
            dataset_name=dataset_name,
            student_optimizer_lr=student_optimizer_lr,
            student_optimizer_weight_decay=student_optimizer_weight_decay,
            discriminator_optimizer_lr=discriminator_optimizer_lr,
            discriminator_optimizer_weight_decay=discriminator_optimizer_weight_decay,
            batch_size=batch_size,
            num_workers=num_workers,
            discriminator_update_freq=discriminator_update_freq,
            train_discriminator_logits=train_discriminator_logits,
            train_discriminator_embeddings=train_discriminator_embeddings,
            epochs=epochs,
            seed=seed,
        )

        trainer.train()
        valid_ap = trainer.evaluate(split="valid")
        test_ap = trainer.evaluate(split="test")
        run_results = {
            "experiment_id": f"student_gine_gakd_{dataset_name}_{include_vn_student}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "dataset_name": dataset_name,
            "seed": seed,
            "run": run + 1,
            "train_discriminator_logits": train_discriminator_logits,
            "train_discriminator_embeddings": train_discriminator_embeddings,
            "n_params": numel(trainer.student_model, only_trainable=True),
            "lr": student_optimizer_lr,
            "weight_decay": student_optimizer_weight_decay,
            "discriminator_lr": discriminator_optimizer_lr,
            "discriminator_weight_decay": discriminator_optimizer_weight_decay,
            "batch_size": trainer.batch_size,
            "epochs": trainer.epochs,
            "valid_metric": valid_ap,
            "test_metric": test_ap,
            "metric": metric,
            "discriminator_update_freq": discriminator_update_freq,
            "train_discriminator_logits": train_discriminator_logits,
            "train_discriminator_embeddings": train_discriminator_embeddings,
            "student_model_args": json.dumps(student_model_args),
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
    print(f"\nSummary Statistics:", flush=True)
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
    parser = argparse.ArgumentParser(
        description="Run GINE experiments with or without virtual nodes under GAKD framework"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of runs to run experiments",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["ogbg-molpcba", "ogbg-molhiv"],
        default="ogbg-molpcba",
        help="Name of the dataset to run experiments on",
    )
    # teacher knowledge path
    parser.add_argument(
        "--teacher_knowledge_path",
        type=str,
        default=f"{base_dir}/teacher_knowledge/teacher_knowledge_ogbg-molpcba.tar",
        help="Path to the teacher knowledge",
    )
    parser.add_argument(
        "--student_embedding_dim",
        type=int,
        default=None,
        help="Embedding dimension for the student model, if None, use the dimension of the teacher model",
    )
    parser.add_argument(
        "--student_out_dim",
        type=int,
        default=None,
        help="Output dimension for the student model, if None, use the dimension of the teacher model",
    )
    parser.add_argument(
        "--student_num_layers",
        type=int,
        default=5,
        help="Number of layers for the student model, if None, use the number of layers of the teacher model",
    )
    parser.add_argument(
        "--student_dropout",
        type=float,
        default=0.5,
        help="Dropout rate for the student model",
    )
    parser.add_argument(
        "--student_virtual_node",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to use virtual nodes in the student model",
    )
    parser.add_argument(
        "--student_train_vn_eps",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to train the virtual node epsilon in the student model",
    )
    parser.add_argument(
        "--student_vn_eps",
        type=float,
        default=0.0,
        help="Virtual node epsilon for the student model",
    )
    parser.add_argument(
        "--student_optimizer_lr",
        type=float,
        default=5e-3,
        help="Learning rate for the student model",
    )
    parser.add_argument(
        "--student_optimizer_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for the student model",
    )
    parser.add_argument(
        "--discriminator_optimizer_lr",
        type=float,
        default=1e-2,
        help="Learning rate for the discriminator",
    )
    parser.add_argument(
        "--discriminator_optimizer_weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for the discriminator",
    )
    parser.add_argument(
        "--discriminator_update_freq",
        type=int,
        default=1,
        help="Frequency of discriminator updates",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the student model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )

    parser.add_argument(
        "--train_discriminator_logits",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to train the discriminator on logits",
    )
    parser.add_argument(
        "--train_discriminator_embeddings",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to train the discriminator on embeddings",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output file",
    )

    args = parser.parse_args()
    virtual_node = args.student_virtual_node.lower() == "true"
    student_args = {
        "embedding_dim": args.student_embedding_dim,
        "out_dim": args.student_out_dim,
        "num_layers": args.student_num_layers,
        "dropout": args.student_dropout,
        "virtual_node": virtual_node,
        "train_vn_eps": args.student_train_vn_eps.lower() == "true",
        "vn_eps": args.student_vn_eps,
    }
    os.makedirs(f"{base_dir}/results", exist_ok=True)
    experiment_type = "with" if virtual_node else "without"
    print(
        f"Running experiments {experiment_type} Virtual Nodes for {args.dataset_name}, Train discriminator logits={args.train_discriminator_logits}, Train discriminator embeddings={args.train_discriminator_embeddings}",
        flush=True,
    )
    if args.output_file is None:
        file_name = f"{base_dir}/results/gine_student_gakd_{args.dataset_name}_{experiment_type}_virtual_node_discriminator_logits_{args.train_discriminator_logits}_discriminator_embeddings_{args.train_discriminator_embeddings}_k{args.discriminator_update_freq}_wd{args.student_optimizer_weight_decay}_drop{args.student_dropout}.csv"
    else:
        file_name = args.output_file

    results_df = run_multiple_experiments(
        args.teacher_knowledge_path,
        args.dataset_name,
        n_runs=args.n_runs,
        student_optimizer_lr=args.student_optimizer_lr,
        student_optimizer_weight_decay=args.student_optimizer_weight_decay,
        discriminator_optimizer_lr=args.discriminator_optimizer_lr,
        discriminator_optimizer_weight_decay=args.discriminator_optimizer_weight_decay,
        discriminator_update_freq=args.discriminator_update_freq,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_file=file_name,
        train_discriminator_logits=args.train_discriminator_logits,
        train_discriminator_embeddings=args.train_discriminator_embeddings,
        student_model_args=student_args,
    )

    print(results_df.to_string(), flush=True)
    print("Experiments completed successfully!", flush=True)
