import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from data_utils import load_ogb_molpcba_dataset, load_teacher_knowledge
# or from data_utils import slice_teacher_train_data if you want to slice teacher knowledge
from models import MLP
from kd_module import MuGSI_MLP_Student

def run_mugsi_mlp_kd(
    data_root,
    teacher_kd_path,
    node_dim=9,       # typical ogbg-molpcba node feature size
    hidden_dim=64,
    num_classes=128,  # for molpcba
    pooling_method='sum',  # 'sum' or 'attention'
    lr=1e-3,
    weight_decay=1e-5,
    max_epochs=30,
    batch_size=32,
    softLabelReg=0.1,
    nodeSimReg=1e-3,
    graphPoolingReg=1e-3,
    randomWalkReg=1e-3,
    device="cuda:0",
    sample_size=20,
    path_length=15
):
    # 1. Load dataset
    train_set, valid_set, test_set = load_ogb_molpcba_dataset(
        root=data_root, 
        sample_size=sample_size, 
        path_length=path_length
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    # 2. Load teacher knowledge
    device_obj = torch.device(device)
    teacher_ptr, teacher_logits, teacher_h, teacher_g = load_teacher_knowledge(teacher_kd_path, device_obj)

    # 3. Build MLP Student
    student_model = MLP(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pooling_method=pooling_method
    ).to(device_obj)

    # 4. Build MuGSI MLP Lightning module
    pl_module = MuGSI_MLP_Student(
        student_model=student_model,
        teacher_ptr=teacher_ptr,
        teacher_logits=teacher_logits,
        teacher_h=teacher_h,
        teacher_g=teacher_g,
        lr=lr,
        weight_decay=weight_decay,
        use_soft_label=True,
        use_nodeSim=True,
        use_graphPooling=True,
        use_randomWalk=True,
        softLabelReg=softLabelReg,
        nodeSimReg=nodeSimReg,
        graphPoolingReg=graphPoolingReg,
        randomWalkReg=randomWalkReg,
        sample_size=sample_size,
        path_length=path_length,
        num_tasks=num_classes
    )

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if "cuda" in device else "cpu",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False
    )

    # 6. Fit
    trainer.fit(pl_module, train_loader, valid_loader)

    # 7. Test
    trainer.test(pl_module, test_loader)


if __name__ == "__main__":
    data_root = "./dataset/ogbg-molpcba-kd-new" 
    teacher_kd_path = "./../../GraphGPS/teacher_results/teacher-knowledge.pt"
    # "./../../GraphGPS/teacher_results/new-teacher-knowledge.pt"

    run_mugsi_mlp_kd(
        data_root=data_root,
        teacher_kd_path=teacher_kd_path,
        node_dim=9,
        hidden_dim=200,
        num_classes=128,
        pooling_method='sum',
        lr=1e-3,
        weight_decay=1e-5,
        max_epochs=30,
        batch_size=256,
        softLabelReg=0.1,
        nodeSimReg=1e-3,
        graphPoolingReg=1e-3,
        randomWalkReg=1e-3,
        device="cuda:0",
        sample_size=20,
        path_length=15
    )
