import json
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import JaccardIndex

from dino import DINOV2EncoderLoRA, load_voc_dataloader


def validate_epoch(dino_lora, val_loader, criterion, f_iou, metrics):
    val_loss = 0.0
    val_iou = 0.0

    dino_lora.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().cuda()
            masks = masks.float().cuda()

            logits = dino_lora(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            y_hat = torch.sigmoid(logits)
            iou_score = f_iou(y_hat, torch.argmax(masks, dim=1).int())
            val_iou += iou_score.item()

    metrics["val_loss"].append(val_loss / len(val_loader))
    metrics["val_iou"].append(val_iou / len(val_loader))


def finetune_dino(config, encoder):
    dino_lora = DINOV2EncoderLoRA(
        encoder=encoder,
        r=config.r,
        n=config.n,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
        n_classes=config.n_classes,
        use_lora=config.use_lora,
    ).cuda()

    train_loader, val_loader = load_voc_dataloader(
        img_dim=config.img_dim, batch_size=config.batch_size
    )

    # Finetuning for segmentation
    criterion = nn.BCEWithLogitsLoss().cuda()
    f_iou = JaccardIndex(task="multiclass", num_classes=config.n_classes).cuda()
    optimizer = optim.AdamW(dino_lora.parameters(), lr=config.lr)

    # Log training and validation metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
    }

    for epoch in range(config.epochs):
        dino_lora.train()

        for images, masks in train_loader:
            images = images.float().cuda()
            masks = masks.float().cuda()

            optimizer.zero_grad()

            logits = dino_lora(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            validate_epoch(dino_lora, val_loader, criterion, f_iou, metrics)
            logging.info(
                f"Epoch: {epoch} - val IoU: {metrics["val_iou"][-1]} "
                f"- val loss {metrics["val_loss"][-1]}"
            )

    # Log metrics & save model
    torch.save(dino_lora.state_dict(), f"{config.exp_name}.pt")

    with open(f"{config.exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lora",
        help="Experiment name",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="loRA rank parameter r",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="large",
        help="DINOv2 backbone parameter [small, base, large, giant]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Finetuning batch size",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=21,
        help="Number of classes",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA",
    )
    parser.add_argument(
        "--img_dim",
        type=int,
        nargs=2,
        default=(490, 490),
        help="Image dimensions (width height)",
    )

    # Decoder parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    config = parser.parse_args()

    # All backbone sizes and configurations
    backbones = {
        "small": "vits14_reg",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    intermediate_layers = {
        "small": [2, 5, 8, 11],
        "base": [2, 5, 8, 11],
        "large": [4, 11, 17, 23],
        "giant": [9, 19, 29, 39],
    }
    embedding_dims = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }
    config.emb_dim = embedding_dims[config.size]

    # TODO: Only with multiscale decoder heads
    config.n = intermediate_layers[config.size]

    encoder = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=f"dinov2_{backbones[config.size]}",
    ).cuda()

    finetune_dino(config, encoder)
