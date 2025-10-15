# utils.py
from typing import Optional, Dict
import torch

# =========================
#  IoU medio (tu función, sin cambios)
# =========================
def compute_iou_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> float:
    """Compute the Intersection over Union metric for the predictions and labels.

    Args:
        y_hat (torch.Tensor): The prediction of dimensions (B, C, H, W), C being
            equal to the number of classes.
        y (torch.Tensor): The label for the prediction of dimensions (B, H, W)
        ignore_index (int | None, optional): ignore label to omit predictions in
            given region.
        eps (float, optional): To smooth the division and prevent division
        by zero. Defaults to 1e-6.

    Returns:
        float: The mean IoU
    """
    num_classes = int(y.max().item() + 1)
    y_hat = torch.argmax(y_hat, dim=1)
    
    ious = []
    for c in range(num_classes):
        y_hat_c = (y_hat == c)
        y_c = (y == c)

        # Ignore all regions with ignore
        if ignore_index is not None:
            mask = (y != ignore_index)
            y_hat_c = y_hat_c & mask
            y_c = y_c & mask

        intersection = (y_hat_c & y_c).sum().float()
        union = (y_hat_c | y_c).sum().float()

        if union > 0:
            ious.append((intersection + eps) / (union + eps))

    return torch.mean(torch.stack(ious))

# =========================
#  Dice medio (mismo patrón que el IoU)
# =========================
@torch.no_grad()
def compute_dice_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> float:
    """Mean Dice across present classes (mismo criterio de presencia que IoU)."""
    num_classes = int(y.max().item() + 1)
    y_pred = torch.argmax(y_hat, dim=1)  # (B,H,W)

    dices = []
    for c in range(num_classes):
        pred_c = (y_pred == c)
        true_c = (y == c)

        if ignore_index is not None:
            mask = (y != ignore_index)
            pred_c = pred_c & mask
            true_c = true_c & mask

        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & (~true_c)).sum().float()
        fn = ((~pred_c) & true_c).sum().float()

        denom = 2 * tp + fp + fn
        if denom > 0:
            dices.append((2 * tp + eps) / (denom + eps))

    return torch.mean(torch.stack(dices)).item() if len(dices) > 0 else float("nan")


# =========================
#  Métricas por clase: Dice e IoU
# =========================
@torch.no_grad()
def per_class_dice_iou(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, float]]:
    """
    Devuelve un dict:
        { class_idx: {"Dice": float|nan, "IoU": float|nan} }
    """
    num_classes = int(y.max().item() + 1)
    y_pred = torch.argmax(y_hat, dim=1)  # (B,H,W)

    out: Dict[int, Dict[str, float]] = {}
    if ignore_index is not None:
        valid = (y != ignore_index)
    else:
        valid = torch.ones_like(y, dtype=torch.bool)

    for c in range(num_classes):
        pred_c = (y_pred == c) & valid
        true_c = (y == c) & valid

        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & (~true_c)).sum().float()
        fn = ((~pred_c) & true_c).sum().float()

        # IoU
        denom_iou = tp + fp + fn
        iou = ((tp + eps) / (denom_iou + eps)).item() if denom_iou > 0 else float("nan")

        # Dice
        denom_dice = 2 * tp + fp + fn
        dice = ((2 * tp + eps) / (denom_dice + eps)).item() if denom_dice > 0 else float("nan")

        out[c] = {"Dice": float(dice), "IoU": float(iou)}

    return out


# =========================
#  Promedio de foreground (excluye clase 0)
# =========================
@torch.no_grad()
def foreground_mean_dice_iou(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Promedio (macro) sobre clases de foreground (1..C-1) para Dice e IoU.
    Asume background = 0.
    """
    per_cls = per_class_dice_iou(y_hat, y, ignore_index, eps)
    # Tomamos todas las clases exceptuando 0 (si existe)
    keys = [k for k in per_cls.keys() if k != 0]
    if len(keys) == 0:
        return {"Dice": float("nan"), "IoU": float("nan")}

    dice_vals = torch.tensor([per_cls[k]["Dice"] for k in keys], dtype=torch.float32)
    iou_vals  = torch.tensor([per_cls[k]["IoU"]  for k in keys], dtype=torch.float32)

    # Ignora NaNs al promediar
    dice_mean = torch.nanmean(dice_vals).item() if torch.any(~torch.isnan(dice_vals)) else float("nan")
    iou_mean  = torch.nanmean(iou_vals).item()  if torch.any(~torch.isnan(iou_vals))  else float("nan")
    return {"Dice": float(dice_mean), "IoU": float(iou_mean)}
