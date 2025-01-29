import torch
import torch.nn.functional as F

def calc_KL_divergence(student_logits, teacher_logits):
    """KL divergence between (log-softmax of student) and (softmax of teacher)."""
    log_p_stu = F.log_softmax(student_logits, dim=1)
    p_tea = F.softmax(teacher_logits, dim=1)
    kl_div = F.kl_div(log_p_stu, p_tea, reduction='batchmean')
    if torch.isnan(kl_div) or torch.isinf(kl_div):
        kl_div = torch.tensor(0.0, device=student_logits.device)
    return kl_div

def calc_node_similarity(node_emb_stu, node_emb_teacher):
    """
    Node-sim: difference of Gram matrices 
    i.e. Fro norm of (stu_emb * stu_emb^T - tea_emb * tea_emb^T).
    """
    X = node_emb_stu @ node_emb_stu.T
    Y = node_emb_teacher @ node_emb_teacher.T
    diff = X - Y
    return torch.norm(diff, p='fro')

def nodeFeatureAlignment(stuNode, teacherNode):
    """
    Simple L2 alignment of normalized features. 
    """
    stuNode_norm = torch.norm(stuNode, dim=-1, keepdim=True) + 1e-12
    teaNode_norm = torch.norm(teacherNode, dim=-1, keepdim=True) + 1e-12

    stuNode_normalized = stuNode / stuNode_norm
    teaNode_normalized = teacherNode / teaNode_norm

    return F.mse_loss(stuNode_normalized, teaNode_normalized)

def calculate_conditional_probabilities(index_arr, feature_matrix):
    """
    For random-walk consistency: 
    We treat index_arr[:,0] as the "source" node, 
    and the subsequent columns as path nodes to measure cond probs.
    """
    first_node_indices = index_arr[:, 0]
    first_node_features = feature_matrix[first_node_indices]  # (num_paths, d)

    remaining_node_indices = index_arr[:, 1:]                # (num_paths, path_len)
    remaining_node_features = feature_matrix[remaining_node_indices]  # (num_paths, path_len, d)

    # Dot product between each source embedding and each step
    dot_products = (first_node_features.unsqueeze(1) * remaining_node_features).sum(dim=-1)
    # shape = (num_paths, path_len)
    return F.softmax(dot_products, dim=-1)


def calculate_kl_loss(stuCondProb, teacherCondProb):
    """
    Computes KL(teacher || student) using PyTorch's F.kl_div for numerical stability.
    """
    # Ensure no log(0) by adding a small epsilon
    epsilon = 1e-12
    teacherCondProb = teacherCondProb + epsilon
    stuCondProb = stuCondProb + epsilon

    # Compute log(studentCondProb)
    log_stu = torch.log(stuCondProb)

    # Compute KL divergence: KL(teacher || student)
    kl_div = F.kl_div(input=log_stu, target=teacherCondProb, reduction='batchmean')

    # Handle potential NaN or Inf
    kl_div = torch.where(torch.isnan(kl_div) | torch.isinf(kl_div), torch.tensor(0.0, device=kl_div.device), kl_div)

    return kl_div


def calculate_kl_loss(pred, target):
    # Calculate the KL divergence for each row
    kl_val = F.kl_div(input=torch.log(pred), target=target, reduction='batchmean')
    # Replace NaN or Inf with zero
    if torch.isnan(kl_val) or torch.isinf(kl_val):
        kl_val = 0.0
    return kl_val
