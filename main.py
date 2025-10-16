import json
import logging
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import tqdm
from dino_finetune import (
    DINOEncoderLoRA,
    get_dataloader,
    visualize_overlay,
    compute_iou_metric,
)
from dino_finetune.metrics import (
    compute_dice_metric,
    foreground_mean_dice_iou,
    per_class_dice_iou,
)


def _log_val_dict(prefix: str, d: dict, step: int):
    flat = {f"{prefix}/{k}": float(v) for k, v in d.items()}
    wandb.log(flat, step=step)  # aquí step será 'epoch'


def validate_epoch(
    dino_lora: nn.Module,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    metrics: dict,
) -> None:
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_iou_fg = 0.0
    val_dice_fg = 0.0
    val_iou_per_class = {}
    val_dice_per_class = {}

    dino_lora.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(val_loader, desc="Valid", unit="batch", leave=False)
        for images, masks in pbar:
            # images = images.float().cuda()
            # masks = masks.long().cuda()
            images = (
                images.cuda(non_blocking=True)
                .to(memory_format=torch.channels_last)
                .float()
            )
            masks = masks.cuda(non_blocking=True).long()
            logits = dino_lora(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            y_hat = torch.sigmoid(logits)
            iou_score = compute_iou_metric(y_hat, masks, ignore_index=255)
            # logits: (B, C, H, W) ; labels: (B, H, W)
            dice_score = compute_dice_metric(logits, masks, ignore_index=255)

            per_cls = per_class_dice_iou(
                logits, masks, ignore_index=255
            )  # {0:{Dice,IoU}, 1:{...}, ...}
            fg_mean = foreground_mean_dice_iou(
                logits, masks, ignore_index=255
            )  # {"Dice":..., "IoU":...}

            val_iou += iou_score.item()
            val_dice += dice_score
            val_iou_fg += fg_mean["IoU"]
            val_dice_fg += fg_mean["Dice"]
            for c, v in per_cls.items():
                if c not in val_iou_per_class:
                    val_iou_per_class[c] = []
                    val_dice_per_class[c] = []
                val_iou_per_class[c].append(v["IoU"])
                val_dice_per_class[c].append(v["Dice"])

            pbar.set_postfix(loss=f"{loss.item():.4f}")
    metrics["val_loss"].append(val_loss / len(val_loader))
    metrics["val_iou"].append(val_iou / len(val_loader))
    metrics["val_iou_fg"].append(val_iou_fg / len(val_loader))
    metrics["val_dice_fg"].append(val_dice_fg / len(val_loader))
    metrics["val_dice"].append(val_dice / len(val_loader))
    metrics["val_iou_per_class"].append(
        {c: sum(v) / len(v) for c, v in val_iou_per_class.items()}
    )
    metrics["val_dice_per_class"].append(
        {c: sum(v) / len(v) for c, v in val_dice_per_class.items()}
    )


def finetune_dino(config: argparse.Namespace, encoder: nn.Module):
    dino_lora = DINOEncoderLoRA(
        encoder=encoder,
        r=config.r,
        emb_dim=config.emb_dim,
        img_dim=config.img_dim,
        n_classes=config.n_classes,
        use_lora=config.use_lora,
        use_fpn=config.use_fpn,
        use_mask2former=config.use_mask2former,
    ).cuda()
    dino_lora = dino_lora.to(memory_format=torch.channels_last)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if config.lora_weights:
        dino_lora.load_parameters(config.lora_weights)

    train_loader, val_loader = get_dataloader(
        config.dataset,
        folder_name=config.folder_name,
        img_dim=config.img_dim,
        batch_size=config.batch_size,
        root=config.root,
        split_json=config.split_json,
        fold=config.fold,
        n_workers=config.n_workers,
    )
    # Finetuning for segmentation
    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
    optimizer = optim.AdamW(
        dino_lora.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Scheduler start warm-up with steady incline, at config.warmup_epochs, start cosine annealing
    warmup_sched = LambdaLR(
        optimizer, lambda epoch: min(1.0, (epoch + 1) / config.warmup_epochs)
    )
    cos_sched = CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=config.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cos_sched],
        milestones=[config.warmup_epochs],
    )

    # Log training and validation metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_iou_fg": [],
        "val_dice_fg": [],
        "val_iou_per_class": [],
        "val_dice_per_class": [],
        "val_dice": [],
    }
    logging.info("Starting training...")

    # check bf16
    torch.backends.cudnn.benchmark = True  # perf
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype is torch.float16))

    for epoch in range(config.epochs):
        dino_lora.train()
        running = 0.0
        pbar = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch"
        )
        for b_idx, (images, masks) in enumerate(pbar):
            # images = images.float().cuda()
            # masks = masks.long().cuda()

            images = (
                images.cuda(non_blocking=True)
                .to(memory_format=torch.channels_last)
                .float()
            )
            masks = masks.cuda(
                non_blocking=True
            ).long()  # máscaras quedan en formato “normal”

            optimizer.zero_grad()

            with torch.autocast("cuda", dtype=dtype):
                logits = dino_lora(images)
                loss = criterion(logits, masks)
            if dtype is torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # logits = dino_lora(images)
            # loss = criterion(logits, masks)
            # loss.backward()
            # optimizer.step()

            running = 0.9 * running + 0.1 * loss.item() if running else loss.item()
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{running:.4f}", lr=f"{lr:.2e}")
            if config.wandb and (b_idx % config.log_interval == 0):
                global_step = epoch * len(train_loader) + b_idx
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/loss_ewm": float(running),
                        "train/lr": float(lr),
                        "train/epoch": epoch,
                        "global_step": global_step,  # <- clave: publica el contador
                    },
                    step=global_step,
                )

        scheduler.step()

        if epoch % 5 == 0:
            y_hat = torch.sigmoid(logits)
            validate_epoch(dino_lora, val_loader, criterion, metrics)

            logging.info(
                f"Epoch: {epoch} - val IoU: {metrics['val_iou'][-1]:.4f} "
                f"- val loss {metrics['val_loss'][-1]:.4f}"
                f"- val IoU FG: {metrics['val_iou_fg'][-1]:.4f}"
                f"- val Dice FG: {metrics['val_dice_fg'][-1]:.4f}"
                f"- val Dice: {metrics['val_dice'][-1]:.4f}"
                f"- val IoU per class: {metrics['val_iou_per_class'][-1]}"
                f"- val Dice per class: {metrics['val_dice_per_class'][-1]}"
            )
            if config.wandb:
                wandb.log(
                    {
                        "val/loss": float(metrics["val_loss"][-1]),
                        "val/iou": float(metrics["val_iou"][-1]),
                        "val/dice": float(metrics["val_dice"][-1]),
                        "val/iou_fg": float(metrics["val_iou_fg"][-1]),
                        "val/dice_fg": float(metrics["val_dice_fg"][-1]),
                        "epoch": epoch,  # <- publica epoch
                    },
                    step=global_step,  # <- step = epoch
                )

                # Por-clase
                _log_val_dict(
                    "val/iou_per_class", metrics["val_iou_per_class"][-1], global_step
                )
                _log_val_dict(
                    "val/dice_per_class", metrics["val_dice_per_class"][-1], global_step
                )

        if epoch % 25 == 0:
            if config.debug:
                visualize_overlay(
                    images,
                    y_hat,
                    config.n_classes,
                    filename=f"output/viz_{config.exp_name}_{epoch}",
                )
            dino_lora.save_parameters(f"output/{config.exp_name}_e{epoch}.pt")

    # Log metrics & save model the final values
    # Saves only loRA parameters and classifer
    dino_lora.save_parameters(f"output/{config.exp_name}.pt")
    with open(f"output/{config.exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f)
    if config.wandb:
        # Sube el JSON de métricas también
        wandb.save(f"output/{config.exp_name}_metrics.json")
        final_art = wandb.Artifact(f"{config.exp_name}-final", type="model")
        final_art.add_file(f"output/{config.exp_name}.pt")
        wandb.log_artifact(final_art)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lora",
        help="Experiment name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug by visualizing some of the outputs to file for a sanity check",
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
        help="DINOv2, DINOv3 backbone parameter [small, base, large, giant]",
    )
    parser.add_argument(
        "--dino_type",
        type=str,
        default="dinov3",
        help="Either [dinov2, dinov3], defaults to DINOv3",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use Low-Rank Adaptation (LoRA) to finetune",
    )
    parser.add_argument(
        "--use_fpn",
        action="store_true",
        help="Use the FPN decoder for finetuning",
    )
    parser.add_argument(
        "--use_mask2former",
        action="store_true",
        help="Use the Mask2Former decoder for finetuning",
    )
    parser.add_argument(
        "--img_dim",
        type=int,
        nargs=2,
        default=(480, 480),
        help="Image dimensions (height width)",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Load the LoRA weights from file location",
    )

    # Training parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="liver",
        help="The dataset to finetune on, either `voc` or `ade20k`",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=20,
        help="Number of epochs of the training epochs for which we do a warm-up.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=3e-5,
        help="lowest learning rate for the scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="The weight decay parameter",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Finetuning batch size",
    )
    parser.add_argument(
        "--folder_name", type=str, default="Dataset771_livervsi"
    )  # <- nuevo
    parser.add_argument(
        "--root", type=str, default="/home/exx/Documents/nnUNetFrame/dataset/nnUNet_raw"
    )
    parser.add_argument("--n_workers", type=int, default=32)
    parser.add_argument("--split_json", type=str, default="./data/splits_final.json")
    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--wandb_project", type=str, default="dinov3-finetune")
    parser.add_argument(
        "--log_interval", type=int, default=50, help="batches between wandb logs"
    )

    config = parser.parse_args()

    # Dataset configuration -> for liver I have 0 background, 1 liver, 2 kidney
    dataset_classes = {"voc": 21, "ade20k": 150, "liver": 3}
    config.n_classes = dataset_classes[config.dataset]

    # Model configuration
    config.patch_size = 16 if config.dino_type == "dinov3" else 14
    backbones = {
        "small": f"{config.dino_type}_vits{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "base": f"{config.dino_type}_vitb{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "large": f"{config.dino_type}_vitl{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "giant": f"{config.dino_type}_vitg{config.patch_size}{'_reg' if config.dino_type == 'dinov2' else ''}",
        "huge": f"{config.dino_type}_vith{config.patch_size}{'plus' if config.dino_type == 'dinov3' else ''}{'_reg' if config.dino_type == 'dinov2' else ''}",
    }

    # encoder = torch.hub.load(
    #     repo_or_dir=f"facebookresearch/{config.dino_type}",
    #     model=backbones[config.size],
    # ).cuda()

    if config.root == "/home/exx/Documents/nnUNetFrame/dataset/nnUNet_raw":
        encoder = torch.hub.load(
            "/data/GitHub/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="/data/GitHub/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        ).cuda()
    else:
        encoder = torch.hub.load(
            "/scratch/bcastane_lab/eochoaal/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="/scratch/bcastane_lab/eochoaal/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        ).cuda()

    if config.wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.exp_name,
            config=vars(config),
        )

        # 1) Nombra las métricas de paso
        wandb.define_metric("global_step")  # contador de batches
        wandb.define_metric("epoch")  # contador de épocas

        # 2) Asocia cada grupo a su step correspondiente
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="epoch")

    config.emb_dim = encoder.num_features

    logging.basicConfig(
        level=logging.INFO,  # muestra INFO+
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),  # consola
            logging.FileHandler(
                f"output/train_fold{config.fold}_bs{config.batch_size}.log", mode="a"
            ),  # archivo
        ],
    )

    if (
        config.img_dim[0] % config.patch_size != 0
        or config.img_dim[1] % config.patch_size != 0
    ):
        logging.info(
            f"The image size ({config.img_dim}) should be divisible "
            f"by the patch size {config.patch_size}."
        )
        # subtract the difference from image size and set a new size.
        config.img_dim = (
            config.img_dim[0] - config.img_dim[0] % config.patch_size,
            config.img_dim[1] - config.img_dim[1] % config.patch_size,
        )
        logging.info(f"The image size is lowered to ({config.img_dim}).")
    finetune_dino(config, encoder)
    if config.wandb:
        wandb.finish()
