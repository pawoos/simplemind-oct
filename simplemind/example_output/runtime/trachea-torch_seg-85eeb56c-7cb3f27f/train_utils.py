"""
train_utils.py
==============

Shared training utilities for SimpleMind segmentation learners (2D & 3D).

Includes:
    - compute_iou
    - plot_loss_miou
    - train_val_test_split
    - add_decoder_dropout
    - is_no_augmentation_config
    - visualization helpers (optional)
"""

import os, re, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn


# -------------------------------------------------------------------------
#  METRICS
# -------------------------------------------------------------------------
def compute_iou(preds, targets, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Compute mean IoU for binary segmentation across batch.

    Args:
        preds (torch.Tensor): raw model outputs (logits)
        targets (torch.Tensor): binary mask (float or int)
        threshold (float): probability cutoff
    """
    preds = torch.sigmoid(preds) > threshold
    targets = targets > 0.5
    dims = tuple(range(1, preds.ndim))
    inter = (preds & targets).float().sum(dim=dims)
    union = (preds | targets).float().sum(dim=dims)
    return ((inter + eps) / (union + eps)).mean().item()


# -------------------------------------------------------------------------
#  TRAIN / VAL / TEST SPLIT
# -------------------------------------------------------------------------
def train_val_test_split(results, split_ratio=(6, 2, 2), seed: int = 42):
    """
    Split a list of SMImage objects into train / val / test lists.
    """
    train_split, val_split, test_split = split_ratio
    test_size_1 = (val_split + test_split) / (train_split + val_split + test_split)
    test_size_2 = test_split / (val_split + test_split)
    train_list, val_list = train_test_split(results, test_size=test_size_1, random_state=seed)
    val_list, test_list = train_test_split(val_list, test_size=test_size_2, random_state=seed)
    return train_list, val_list, test_list


# -------------------------------------------------------------------------
#  LOSS / METRIC PLOTTING
# -------------------------------------------------------------------------
def plot_loss_miou(log_path, plot_path):
    """
    Parse the training log and plot Train/Val losses, components, and Val mIoU.

    Parameters
    ----------
    log_path : str
        Path to the 'log_training.txt' file written during training.
    plot_path : str
        Path to save the resulting PNG figure.

    Output
    ------
    Saves a combined plot with:
        - Train Loss
        - Val Loss
        - Val Dice Loss
        - Val CE Loss
        - Val mIoU (secondary axis)
    """
    train_losses, val_losses = [], []
    val_dice_losses, val_ce_losses, val_mious = [], [], []

    with open(log_path, "r") as f:
        for line in f:
            t = re.search(r"Train Loss:\s*([0-9.]+)", line)
            v = re.search(r"Validation Loss:\s*([0-9.]+)", line)
            vd = re.search(r"Validation Dice Loss:\s*([0-9.]+)", line)
            vc = re.search(r"Validation CE Loss:\s*([0-9.]+)", line)
            m = re.search(r"Validation mIoU:\s*([0-9.]+)", line)
            if t:
                train_losses.append(float(t.group(1)))
            if v:
                val_losses.append(float(v.group(1)))
            if vd:
                val_dice_losses.append(float(vd.group(1)))
            if vc:
                val_ce_losses.append(float(vc.group(1)))
            if m:
                val_mious.append(float(m.group(1)))

    if len(train_losses) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    def _plot_series(ax, values, label, marker, color):
        if not values:
            return
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=label, marker=marker, color=color)

    _plot_series(ax1, train_losses, "Train Loss", 'o', 'tab:blue')
    _plot_series(ax1, val_losses, "Val Loss", 's', 'tab:orange')
    _plot_series(ax1, val_dice_losses, "Val Dice Loss", '^', 'tab:green')
    _plot_series(ax1, val_ce_losses, "Val CE Loss", 'd', 'tab:red')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    if val_mious:
        epochs_miou = range(1, len(val_mious) + 1)
        ax2.plot(epochs_miou, val_mious, label="Val mIoU", marker='x', color='tab:purple')
        ax2.set_ylabel("Validation mIoU", color='tab:purple')
        ax2.tick_params(axis='y', labelcolor='tab:purple')

    lines, labels = ax1.get_legend_handles_labels()
    if val_mious:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="center right")
    else:
        ax1.legend(loc="upper right")

    plt.title("Training & Validation Loss Components with Val mIoU")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)



# -------------------------------------------------------------------------
#  VISUALIZATION UTILITIES (homogeneous across 2D and 3D)
# -------------------------------------------------------------------------
def _normalize_image(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def overlay_mask(img, mask, alpha=0.4):
    """
    Overlay red/yellow mask on grayscale image:
      - Red for mask >= 0.8
      - Yellow for 0.5 <= mask < 0.8
    """
    img = _normalize_image(img)
    mask = np.clip(mask, 0, 1)
    rgb = np.stack([img]*3, axis=-1)

    overlay = np.zeros_like(rgb)
    red_mask = mask >= 0.8
    yellow_mask = (mask >= 0.5) & (mask < 0.8)
    overlay[..., 0][red_mask | yellow_mask] = 1.0
    overlay[..., 1][yellow_mask] = 1.0

    return np.clip(rgb*(1-alpha) + overlay*alpha, 0, 1)


def visualize_sample_2d(img, pred, label, save_path, miou=None, title_prefix=""):
    """
    Save 2D input / prediction / ground truth comparison.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img, pred, label = _normalize_image(img), np.clip(pred, 0, 1), np.clip(label, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(overlay_mask(img, pred)); axes[1].set_title("Prediction"); axes[1].axis("off")
    axes[2].imshow(overlay_mask(img, label)); axes[2].set_title("Ground Truth"); axes[2].axis("off")

    if miou is not None:
        fig.suptitle(f"{title_prefix} mIoU: {miou:.3f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_sample_3d(input_tensor, pred_tensor, label_tensor, save_path, miou=None, title_prefix="", centroid=True):
    """
    Visualize orthogonal slices (axial/coronal/sagittal) for a 3D volume.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = input_tensor[0].detach().cpu().numpy()
    pred = torch.sigmoid(pred_tensor[0]).detach().cpu().numpy() if torch.is_tensor(pred_tensor) else pred_tensor[0]
    label = label_tensor[0].detach().cpu().numpy() if torch.is_tensor(label_tensor) else label_tensor[0]

    # --- pick slice indices ---
    coords = np.argwhere(label > 0.5)
    if len(coords) == 0:
        zc, yc, xc = np.array(label.shape)//2
    else:
        zc, yc, xc = np.round(coords.mean(axis=0)).astype(int)

    slices = {
        "axial":   (img[zc,:,:], pred[zc,:,:], label[zc,:,:]),
        "coronal": (img[:,yc,:], pred[:,yc,:], label[:,yc,:]),
        "sagittal":(img[:,:,xc], pred[:,:,xc], label[:,:,xc])
    }

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for r, (plane, (i_img, p_img, l_img)) in enumerate(slices.items()):
        axes[r,0].imshow(i_img, cmap="gray");  axes[r,0].set_title(f"Input ({plane})"); axes[r,0].axis("off")
        axes[r,1].imshow(overlay_mask(i_img, p_img)); axes[r,1].set_title("Prediction"); axes[r,1].axis("off")
        axes[r,2].imshow(overlay_mask(i_img, l_img)); axes[r,2].set_title("Ground Truth"); axes[r,2].axis("off")

    if miou is not None or title_prefix:
        prefix = f"{title_prefix} " if title_prefix else ""
        miou_str = f"mIoU = {miou:.3f}" if miou is not None else ""
        fig.suptitle(f"{prefix}{miou_str}".strip(), fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
