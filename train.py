import os
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from train_loader import DinoV3TwoViewTripletDataset
from train_model import DinoV3TripletModel
from train_loss import BatchHardTripletLoss
from utils import log_print, save_checkpoint
from config import parse_args, build_config


def validate(model, val_loader, criterion, device):
    """
    Compute average loss on validation set.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Torch device (cuda/cpu)

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            v1 = batch["pixel_values1"].to(device)
            v2 = batch["pixel_values2"].to(device)
            labels = batch["labels"].to(device)

            # Combine views
            all_views = torch.cat([v1, v2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)

            # Forward pass
            embeddings = model(all_views)
            loss = criterion(embeddings, all_labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    return avg_loss

def main():
    """
    Main training loop for DINOv3 Triplet Model.
    """
    # Parse command line arguments and build configuration
    args = parse_args()
    config = build_config(args)

    checkpoint_dir = os.path.join(config["output_dir"], "checkpoints")
    log_file = os.path.join(config["output_dir"], config["log_file"])

    # Create output directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Print parameters
    log_print("=" * 50, log_file)
    log_print("========== Training Parameters ==========", log_file)
    log_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print("=" * 50, log_file)
    for key, value in config.items():
        log_print(f"{key:20s}: {value}", log_file)
    log_print("=" * 50, log_file)
    log_print("", log_file)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}", log_file)
    log_print("", log_file)

    # Load processor
    log_print("Loading processor...", log_file)
    processor = AutoImageProcessor.from_pretrained(
        config["model_name"],
        cache_dir=config["model_dir"],
        local_files_only=True
    )

    # Create label maps
    log_print("Creating label maps...", log_file)
    train_categories = os.listdir(config["train_dir"])
    train_label_map = {
        cat_name: idx
        for idx, cat_name in enumerate(sorted(train_categories))
    }
    log_print(f"Train categories: {len(train_categories)}", log_file)

    val_categories = os.listdir(config["valid_dir"])
    val_label_map = {
        cat_name: idx
        for idx, cat_name in enumerate(sorted(val_categories))
    }
    log_print(f"Validation categories: {len(val_categories)}", log_file)
    log_print("", log_file)

    # Create data loaders
    log_print("Creating data loaders...", log_file)
    train_dataset = DinoV3TwoViewTripletDataset(
        config["train_dir"],
        train_label_map,
        processor
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True
    )
    log_print(f"Train dataset size: {len(train_dataset)}", log_file)

    val_dataset = DinoV3TwoViewTripletDataset(
        config["valid_dir"],
        val_label_map,
        processor
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False
    )
    log_print(f"Validation dataset size: {len(val_dataset)}", log_file)
    log_print("", log_file)

    # Create model and optimizer
    log_print("Creating model...", log_file)
    model = DinoV3TripletModel(
        model_name=config["model_name"],
        proj_dim=config["proj_dim"],
        freeze_until=config["freeze_until"],
        model_dir=config["model_dir"]
    ).to(device)

    criterion = BatchHardTripletLoss(margin=config["margin"])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    log_print("Model created successfully", log_file)
    log_print("", log_file)

    # Training loop
    scaler = torch.amp.GradScaler('cuda')
    best_loss = float('inf')

    log_print("Starting training...", log_file)
    log_print("=" * 50, log_file)
    log_print("", log_file)

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            v1 = batch["pixel_values1"].to(device)
            v2 = batch["pixel_values2"].to(device)
            labels = batch["labels"].to(device)

            all_views = torch.cat([v1, v2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                embeddings = model(all_views)
                loss = criterion(embeddings, all_labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        log_print(
            f"Epoch {epoch+1}/{config['num_epochs']} - "
            f"Train loss: {avg_train_loss:.4f}, "
            f"Val loss: {val_loss:.4f}",
            log_file
        )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                path=os.path.join(checkpoint_dir, "dinov3_triplet_best.pth"),
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                best_loss=best_loss,
                log_file=log_file
            )

    # Save final model
    log_print("", log_file)
    save_checkpoint(
        path=os.path.join(checkpoint_dir, "dinov3_triplet_last.pth"),
        epoch=config["num_epochs"],
        model=model,
        optimizer=optimizer,
        best_loss=best_loss,
        log_file=log_file
    )

    log_print("", log_file)
    log_print("=" * 50, log_file)
    log_print("Training completed!", log_file)
    log_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_print(f"Best validation loss: {best_loss:.4f}", log_file)
    log_print("=" * 50, log_file)


if __name__ == "__main__":
    main()
