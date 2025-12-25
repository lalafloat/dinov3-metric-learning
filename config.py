"""
Configuration management for DINOv3 Triplet Model training
"""
import argparse
from typing import Dict, Any


# Default parameters
DEFAULT_PARAMS = {
    # Model settings
    "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "model_dir": "weights",  # Relative to project root
    "proj_dim": 384,
    "freeze_until": 10,

    # Data settings
    "train_dir": "data/train",  # Relative to project root
    "valid_dir": "data/valid",  # Relative to project root

    # Training settings
    "batch_size": 32,
    "num_workers": 4,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "margin": 0.2,
    "grad_clip": 1.0,

    # Output settings
    "log_file": "log.txt",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train DINOv3 Triplet Model")
    parser.add_argument("output_dir", type=str, help="Output directory for checkpoints and logs")

    # Model settings
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--model_dir", type=str, default=None, help="Model directory")
    parser.add_argument("--proj_dim", type=int, default=None, help="Projection dimension")
    parser.add_argument("--freeze_until", type=int, default=None, help="Freeze until layer")

    # Data settings
    parser.add_argument("--train_dir", type=str, default=None, help="Training data directory")
    parser.add_argument("--valid_dir", type=str, default=None, help="Validation data directory")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--margin", type=float, default=None, help="Triplet loss margin")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build configuration dictionary from default params and command line args.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = DEFAULT_PARAMS.copy()
    config["output_dir"] = args.output_dir

    # Override with command line arguments if provided
    for key in DEFAULT_PARAMS.keys():
        arg_value = getattr(args, key, None)
        if arg_value is not None:
            config[key] = arg_value

    return config
