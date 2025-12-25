"""
DINOv3 Triplet Model for metric learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoV3TripletModel(nn.Module):
    """
    DINOv3-based model with projection head for triplet loss training.
    """

    def __init__(self,
                 model_name: str = "facebook/dinov3-vits16",
                 proj_dim: int = 256,
                 freeze_until: int = 10,
                 model_dir: str = None):
        """
        Initialize DINOv3 Triplet Model.

        Args:
            model_name (str): Hugging Face model name for DINOv3
            proj_dim (int): Dimension of final embedding (for FAISS, etc.)
            freeze_until (int): Freeze encoder layers with index < freeze_until.
                               Only train the latter layers and projection head.
            model_dir (str, optional): Directory to cache the pretrained model.
                                      If None, uses default cache directory.
        """
        super().__init__()

        # Load backbone model
        if model_dir:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=True
            )
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size

        # Projection head: CLS token embedding -> triplet embedding
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Freeze early ViT layers and embedding layers
        for name, param in self.backbone.named_parameters():
            if "encoder.layer" in name:
                try:
                    layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                except Exception:
                    layer_idx = 0
                if layer_idx < freeze_until:
                    param.requires_grad = False
            elif any(kw in name for kw in ["embeddings", "patch_embeddings", "position_embeddings"]):
                param.requires_grad = False

    def forward(self, pixel_values):
        """
        Forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input images with shape [B, C, H, W]

        Returns:
            torch.Tensor: L2-normalized embeddings with shape [B, D]
        """
        outputs = self.backbone(pixel_values=pixel_values)
        cls = outputs.pooler_output      # [B, hidden_dim]
        z = self.projection_head(cls)    # [B, proj_dim]
        z = F.normalize(z, p=2, dim=-1)  # L2 normalize
        return z
