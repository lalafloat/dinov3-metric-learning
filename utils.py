"""
Utility functions for training DINOv3 Triplet Model
"""
import torch
from typing import Optional


def log_print(message: str, log_file: Optional[str] = None) -> None:
    """
    Print message to stdout and optionally write to log file.

    Args:
        message (str): Message to print
        log_file (str, optional): Path to log file (if None, only print to stdout)
    """
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    log_file: Optional[str] = None
) -> None:
    """
    Save model checkpoint to disk.

    Args:
        path (str): Path to save checkpoint
        epoch (int): Current epoch number
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        best_loss (float): Best validation loss so far
        log_file (str, optional): Path to log file for logging
    """
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(ckpt, path)
    log_print(f"Saved checkpoint to {path}", log_file)
