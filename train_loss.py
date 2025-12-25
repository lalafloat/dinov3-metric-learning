"""
Batch Hard Triplet Loss implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss with online mining.

    For each anchor, selects the hardest positive (farthest same-class sample)
    and hardest negative (closest different-class sample) within the batch.
    """

    def __init__(self, margin: float = 0.2, p: int = 2):
        """
        Initialize Batch Hard Triplet Loss.

        Args:
            margin (float): Triplet margin
            p (int): Norm for distance calculation (2 = L2 norm)
        """
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, embeddings, labels):
        """
        Compute batch hard triplet loss.

        Args:
            embeddings (torch.Tensor): L2-normalized embeddings with shape [B, D]
            labels (torch.Tensor): Class IDs with shape [B]

        Returns:
            torch.Tensor: Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Compute pairwise squared L2 distance matrix [B, B]
        # dist(i,j) = ||e_i - e_j||_2^2
        # = ||e_i||^2 + ||e_j||^2 - 2 * e_i·e_j
        # For stability, we use squared distance without sqrt
        emb = embeddings.float()

        # Compute dot product and squared distance matrix [B, B]
        dot = torch.matmul(emb, emb.t())
        sq_norm = torch.diag(dot)  # [B]
        dist_sq = sq_norm.unsqueeze(1) - 2 * dot + sq_norm.unsqueeze(0)  # [B, B]
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # Create label masks
        labels = labels.view(-1, 1)  # [B, 1]
        mask_pos = (labels == labels.t())  # [B, B] same class
        mask_neg = (labels != labels.t())  # [B, B] different class

        # Exclude self-comparisons
        diag = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask_pos = mask_pos & (~diag)

        # Find hardest positive: farthest same-class sample
        # Set non-positive distances to 0 (dist >= 0, so this is safe)
        dist_pos = dist_sq.clone()
        dist_pos[~mask_pos] = 0.0
        hardest_pos_dist, _ = dist_pos.max(dim=1)

        # Find hardest negative: closest different-class sample
        # Set non-negative distances to large value to exclude from min
        dist_neg = dist_sq.clone()
        max_dist = dist_sq.max().detach()
        large_val = max_dist + 1.0
        dist_neg[~mask_neg] = large_val
        hardest_neg_dist, _ = dist_neg.min(dim=1)

        # Compute triplet loss: max(0, margin + d_pos - d_neg)
        loss = F.relu(self.margin + hardest_pos_dist - hardest_neg_dist)

        # Exclude anchors without positive samples (single-sample class in batch)
        has_pos = mask_pos.any(dim=1)  # [B]
        if has_pos.any():
            loss = loss[has_pos]
            return loss.mean()
        else:
            # Return zero if no positive pairs exist (should revise batch composition)
            return torch.tensor(0.0, device=device, requires_grad=True)
