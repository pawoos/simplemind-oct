"""
Tool Name: torch3d_segmentation_learn
=====================================

Description:
    Trains a 3D UNet segmentation model using the MONAI framework on volumetric
    medical imaging data (e.g., CT, MRI). Compatible with the SMImageDataset
    class, which provides 4D tensors [C, D, H, W].

    The pipeline supports standard training/validation/test splits, Dice+CE loss,
    early stopping, learning rate scheduling, and optional on-the-fly 3D data
    augmentations via MONAI transforms.

Parameters:
    - input_image (SMImage):
        Input image passed from the pipeline. Each SMImage contains the pixel_array
        and corresponding label_array (segmentation mask).
    - dataset_id (str):
        Unique dataset identifier, used to create output/log directories.
    - results (list[SMImage]):
        List of SMImage objects collected by the aggregator for training.
    - num_classes (int):
        Number of segmentation classes (1 for binary segmentation).
    - learning_rate (float):
        Initial learning rate for Adam optimizer.
    - batch_size (int):
        Batch size used for training and validation.
    - num_epochs (int):
        Number of training epochs.
    - dropout (float, optional):
        Dropout probability applied to UNet decoder blocks to reduce overfitting.
        Default = 0.0 (disabled).
    - weight_decay (float, optional):
        L2 regularization coefficient applied to optimizer. Default = 0.0.
    - factor (float, optional):
        Reduction factor for learning rate when validation performance stalls.
        Used by ReduceLROnPlateau scheduler. Default = 0.5.
    - patience (int, optional):
        Number of epochs without improvement before learning rate is reduced.
        Default = 5.
    - early_stop_patience (int, optional):
        Number of epochs without improvement before training stops early.
        Default = 20.
    - dice_weight / ce_weight (float, optional):
        Relative weights for the Dice vs CE components when num_classes=1 (binary).
        For multi-class training these weights are ignored.
    - loss_include_background (bool, optional):
        Default = False.
        DiceCELoss internally converts the binary mask into two channels (foreground and background) when `loss_include_background=true`.
        For relatively small structures (like kidneys), a separate background channel can dominate the CE term, so excluding it often improves convergence on the organ.
        If background has meaningful structure (multiple tissues) or you worry about false positives, keeping the background channel can help the model learn “what not to segment.”
    - gradient_accumulation (int, optional):
        Accumulate gradients over this many mini-batches before stepping the optimizer,
        which simulates a larger effective batch size when GPU memory is limited.
        When you set gradient_accumulation = k, you process k mini-batches, sum their gradients (each scaled so the final sum matches the average), and only then call optimizer.step(). 
        This makes it equivalent to training with a single batch that's k times larger.
    - split_ratio (list[float, float, float], optional):
        Train / validation / test split proportions. Default = (6, 2, 2).
    - split_seed (int, optional):
        Random seed controlling reproducible dataset splitting. Default = 42.
    - gpu_num (int, optional): gpu_num (int, optional): GPU to be used for computation. Default = 0.
        If GPU not available, falls back to CPU.
    - output_dir (str, optional):
        Output directory for model weights, logs, and sample visualizations.

    # ---- 3D Augmentation Parameters (MONAI) ----
    - aug_h_flip_prob : float, optional
        Probability of flipping the volume along the left–right (X) axis.
        Typical range: 0.0–0.5. Default = 0.0.
    - aug_v_flip_prob : float, optional
        Probability of flipping along the anterior–posterior (Y) axis.
        Typical range: 0.0–0.5. Default = 0.0.
    - aug_affine_p : float, optional
        Probability of applying a 3D affine transformation (rotation + scale).
        Default = 0.0 (disabled).
    - aug_rotate_range : list[float, float, float], optional
        Maximum rotation in radians around each axis (X, Y, Z).
        Example: [0.3, 0.3, 0.3] ≈ ±17°. Default = [0.0, 0.0, 0.0].
    - aug_scale_range : list[float, float, float], optional
        Scaling variation for each axis (fractional).
        Example: [0.1, 0.1, 0.1] allows ±10% zoom per axis.
        Default = [0.0, 0.0, 0.0] (no scaling).
    - aug_deformation_p : float, optional
        Probability of applying a 3D elastic deformation (random smooth warp).
        Default = 0.0 (disabled).
    - aug_deformation_sigma_range : list[float, float], optional
        Range of Gaussian smoothing (σ, in voxels) controlling field smoothness.
        Larger σ → smoother deformation.
        For modest deformation of a 128x128x128 volume: [2, 3].
        Default = [0.0, 0.0].
    - aug_deformation_alpha_range : list[float, float], optional
        Range of displacement magnitudes (α, in voxels) controlling warp strength.
        Larger α → stronger deformations.
        For modest deformation of a 128x128x128 volume: [10, 20].
        Default = [0.0, 0.0].
    - test_sample_percentiles (list[float], optional):
        Percentiles used to pick representative test sample images.
        Default = [10, 25, 26, 49, 50, 51, 75, 76, 90].
    - test_iou_percentiles (list[float], optional):
        Percentiles used to log summary IoU stats in the test section.
        Default = [10, 25, 50, 75, 90].

Output:
    - Writes model checkpoint ("checkpoint.pth"), training log ("log_training.txt"),
      model summary ("log_model.txt"), and optional sample visualizations
      (under `sample_augmentations/` and `epoch_images/`).

Notes:
    - Augmentations are implemented using MONAI’s composable transforms:
        RandFlipd, RandAffined, and Rand3DElasticd.
    - Elastic deformation parameters (`alpha_range`, `sigma_range`, `p`) strongly
      influence visual effect; smaller σ and larger α yield more pronounced warps.
    - Uses Dice + CrossEntropy (DiceCELoss) as the default objective, suitable
      for both binary and multi-class 3D segmentation.
    - Saves best model based on validation mean IoU and supports early stopping.
    - Approximately 5% of augmented samples are automatically visualized to disk
      for QC and debugging.
"""

import os, re, asyncio, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import warnings

from monai.networks.nets import UNet as UNet3D
from sm_sample_aggregator import SMSampleAggregator
from sm_image_dataset import SMImageDataset
from sm_image import SMImage
# from losses import DiceBCELoss
from monai.losses import DiceCELoss, DiceLoss

from train_utils import (
    compute_iou,
    plot_loss_miou,
    train_val_test_split,
    visualize_sample_3d,
)

# Silence noisy framework warnings that clutter stderr but do not affect training.
warnings.filterwarnings("ignore", message=".*verbose parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*single channel prediction.*include_background=False.*")
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")

class Torch3DSegmentationLearn(SMSampleAggregator):

    async def execute(
        self,
        *,
        input_image: SMImage
    ) -> SMImage:
        """
        Accepts an input image and passes it on to the aggregator.
        """   
        return input_image

    async def aggregate(
        self,
        *,
        dataset_id: str,
        results: list[SMImage],
        total: int,
        num_classes: int,
        batch_size: int,
        num_epochs: int,
        gradient_accumulation: int = 1,
        channels: list = [16, 32, 64, 128, 256],
        strides: list = [2, 2, 2, 2],
        loss_dice_weight: float = 1.0,
        loss_ce_weight: float = 1.0,
        loss_include_background: bool = False,
        learning_rate: float = 1e-4,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        aug_h_flip_prob: float = 0.0,
        aug_v_flip_prob: float = 0.0,
        aug_affine_p: float = 0.0,
        aug_rotate_range: list = [0.0, 0.0, 0.0],
        aug_scale_range: list = [0.0, 0.0, 0.0], # Tuple[float, float]
        aug_deformation_p: float = 0.0,
        aug_deformation_sigma_range: list = [0.0, 0.0],
        aug_deformation_alpha_range: list = [0.0, 0.0],
        test_sample_percentiles: list = [10, 25, 26, 49, 50, 51, 75, 76, 90],
        test_iou_percentiles: list = [10, 25, 50, 75, 90],
        early_stop_patience: int = 20,
        split_ratio: list = [6,2,2],
        split_seed: int = 42,
        gpu_num: int = 0,
        output_dir: str = None
    ):
        # ---------- setup ----------
        out_dir = self.resolve_output_dir(output_dir, dataset_id)
        root_dir = self.dataset_output_path(out_dir, self.name().replace(f"-{self.plan_id}", "")) 
        os.makedirs(root_dir, exist_ok=True)
        log_model_file = os.path.join(root_dir, "log_model.txt")
        log_training_file = os.path.join(root_dir, "log_training.txt")

        device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        self.print_log(f"Running MONAI 3D UNet on {device}")

        # Set up augmentation parameters
        augmentation_params = {
            'h_flip_prob': aug_h_flip_prob,
            'v_flip_prob': aug_v_flip_prob,
            'affine_p': aug_affine_p,
            'rotate_range': aug_rotate_range,
            'scale_range': tuple(aug_scale_range),
            'deformation_alpha_range': tuple(aug_deformation_alpha_range), 
            'deformation_sigma_range': tuple(aug_deformation_sigma_range), 
            'deformation_p': aug_deformation_p,
            'vis_dir':root_dir
        }

        # ---------- data ----------
        split_ratio_tuple = tuple(split_ratio) if not isinstance(split_ratio, tuple) else split_ratio
        train_list, val_list, test_list = train_val_test_split(results, split_ratio_tuple, split_seed)
        train_dataset = SMImageDataset(train_list, num_classes=num_classes, augmentation_params=augmentation_params)
        val_dataset   = SMImageDataset(val_list, num_classes=num_classes)
        test_dataset  = SMImageDataset(test_list, num_classes=num_classes)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        with open(log_training_file, 'w') as f:
            f.write(f"num training samples = {len(train_loader.dataset)}\n")        
            f.write(f"num validation samples = {len(val_loader.dataset)}\n") 
            f.write(f"num test samples = {len(test_loader.dataset)}\n") 
            f.write('-' * 50); f.flush()

        # ---------- model ----------
        model = UNet3D(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=2,
            dropout=dropout
        ).to(device)
        with open(log_model_file, 'w') as f:
            f.write(f"{model}")

        # Use MONAI’s combined Dice + CrossEntropy loss (logit-safe)
        criterion = DiceCELoss(
            sigmoid=True,        # applies sigmoid internally
            to_onehot_y=False,   # binary segmentation
            include_background=loss_include_background,
            lambda_dice=loss_dice_weight,
            lambda_ce=loss_ce_weight
        )        
        # criterion = DiceBCELoss()
        dice_loss_fn = DiceLoss(
            sigmoid=True,
            include_background=loss_include_background,
            reduction="mean"
        )
        bce_loss_fn = nn.BCEWithLogitsLoss()
        
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=lr_factor, patience=lr_patience, verbose=False)

        # ---------- training loop ----------
        best_miou, early_stop_counter = 0.0, 0
        gradient_accumulation = max(1, int(gradient_accumulation))
        with open(log_training_file, 'a') as f:  # Open the log file
            for epoch in range(num_epochs):
                free, total = torch.cuda.mem_get_info()
                current_lr = opt.param_groups[0]['lr']
                f.write(f"Epoch {epoch+1}/{num_epochs}\n")
                f.write(f"Available: {free / 1024**2:.2f} MB / {total / 1024**2:.2f} MB\n")
                f.write(f"Learning rate: {current_lr:.6f}\n"); f.flush()

                model.train()
                total_loss = 0.0
                train_dice_total = 0.0
                train_ce_total = 0.0
                opt.zero_grad()
                for step, (images, labels) in enumerate(train_loader, start=1):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    (loss / gradient_accumulation).backward()
                    if step % gradient_accumulation == 0 or step == len(train_loader):
                        opt.step()
                        opt.zero_grad()
                    total_loss += loss.item() * images.size(0)
                    dice_batch = dice_loss_fn(outputs, labels).item()
                    ce_batch = bce_loss_fn(outputs, labels).item()
                    train_dice_total += dice_batch * images.size(0)
                    train_ce_total += ce_batch * images.size(0)
                train_loss = total_loss / len(train_loader.dataset)
                train_dice_loss = train_dice_total / len(train_loader.dataset)
                train_ce_loss = train_ce_total / len(train_loader.dataset)
                f.write(f"Train Loss: {train_loss:.4f}\n"); f.flush()
                f.write(f"Train Dice Loss: {train_dice_loss:.4f}\n")
                f.write(f"Train CE Loss: {train_ce_loss:.4f}\n")

                # ----- validation -----
                model.eval()
                val_loss, val_iou = 0.0, 0.0
                val_dice_total, val_ce_total = 0.0, 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        val_loss += criterion(outputs, labels).item() * images.size(0)
                        val_iou += compute_iou(outputs, labels) * images.size(0)
                        val_dice_total += dice_loss_fn(outputs, labels).item() * images.size(0)
                        val_ce_total += bce_loss_fn(outputs, labels).item() * images.size(0)
                        # val_iou += Torch3DSegmentationLearn.compute_iou(outputs, labels) * images.size(0)
                val_loss /= len(val_loader.dataset)
                val_iou /= len(val_loader.dataset)
                val_dice_loss = val_dice_total / len(val_loader.dataset)
                val_ce_loss = val_ce_total / len(val_loader.dataset)
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write(f"Validation Dice Loss: {val_dice_loss:.4f}\n")
                f.write(f"Validation CE Loss: {val_ce_loss:.4f}\n")
                f.write(f"Validation mIoU: {val_iou:.4f}\n")
                if epoch % 5 == 0:
                    self.save_epoch_images(images, outputs, labels, num_classes, epoch, root_dir, miou=val_iou)
                    
                scheduler.step(val_iou)
                if val_iou > best_miou:
                    best_miou = val_iou
                    early_stop_counter = 0
                    torch.save(model.state_dict(), os.path.join(root_dir, "checkpoint.pth"))
                    f.write(f"Improved! Saved best model (mIoU={best_miou:.4f})\n")
                else:
                    early_stop_counter += 1
                    f.write(f"No improvement ({early_stop_counter}/{early_stop_patience})\n")
                if early_stop_counter >= early_stop_patience:
                    f.write(f"Early stopping after {early_stop_patience} stagnant epochs.\n")
                    break
                
                f.write('-' * 50); f.flush()
                
                plot_file = os.path.join(root_dir, "loss_miou_curve.png")
                plot_loss_miou(log_training_file, plot_file)
                # self.plot_loss_miou(log_training_file, plot_file)

        # ---------- test ----------
        model.load_state_dict(torch.load(os.path.join(root_dir, "checkpoint.pth")))
        model.eval()
        test_loss_total = 0.0
        sample_ious = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # ---- Compute loss ----
                loss = criterion(outputs, labels)
                test_loss_total += loss.item() * images.size(0)

                # Per-sample IoU so logging reflects sample variability
                for i in range(images.size(0)):
                    iou = compute_iou(outputs[i:i+1], labels[i:i+1])
                    # iou = Torch3DSegmentationLearn.compute_iou(outputs[i:i+1], labels[i:i+1])
                    sample_ious.append(iou)

        test_loss = test_loss_total / len(test_loader.dataset)
        mean_iou = float(np.mean(sample_ious)) if sample_ious else 0.0
        # self.print_log(f"Final Test mIoU: {mean_iou:.4f}")

        def percentile_nearest(arr, perc):
            try:
                return np.percentile(arr, perc, method="nearest")
            except TypeError:
                return np.percentile(arr, perc, interpolation="nearest")

        with open(log_training_file, 'a') as f:
            f.write('-' * 50 + "\n")
            f.write(f"Final Test Loss: {test_loss:.4f}\n")
            f.write(f"Final Test mIoU: {mean_iou:.4f}\n")
            log_percentiles = list(test_iou_percentiles) if test_iou_percentiles else []
            if log_percentiles:
                log_vals = percentile_nearest(np.array(sample_ious), log_percentiles) if sample_ious else []
                f.write("Test IoU percentiles:\n")
                for p, val in zip(log_percentiles, np.atleast_1d(log_vals)):
                    f.write(f"  p{p}: {val:.4f}\n")
            else:
                f.write("Test IoU percentiles: not configured; logging all IoUs\n")
                for idx, val in enumerate(sample_ious):
                    f.write(f"  sample_{idx}: {val:.4f}\n")
            f.flush()

        self.save_representative_test_samples(
            test_loader, model, criterion, root_dir, device, test_sample_percentiles
        )


    def save_epoch_images(self, inputs, outputs, labels, num_classes, epoch, root_dir, miou=None):
        """
        Save the axial, coronal, and sagittal planes through the slices
        where the reference label has the most pixels along each axis.
        Each view shows Input | Prediction | Ground Truth.
        """
        os.makedirs(os.path.join(root_dir, "epoch_images"), exist_ok=True)

        inputs_cpu = inputs.detach().cpu()
        outputs_cpu = outputs.detach().cpu()
        labels_cpu = labels.detach().cpu()

        max_samples = min(inputs_cpu.size(0), 2)
        for idx in range(max_samples):
            save_path = os.path.join(
                root_dir,
                "epoch_images",
                f"epoch_{epoch}_sample_{idx}.png",
            )
            visualize_sample_3d(
                inputs_cpu[idx],
                outputs_cpu[idx],
                labels_cpu[idx],
                save_path,
                miou=miou,
                title_prefix=f"Epoch {epoch} | Sample {idx}",
            )


    def save_representative_test_samples(
        self, test_loader, model, criterion, root_dir, device, sample_percentiles
    ):
        """
        Save representative test samples with axial, coronal, and sagittal views.
        Each figure shows Input | Prediction | Ground Truth for the slice with
        the largest reference mask area in that plane.
        """
        os.makedirs(os.path.join(root_dir, "test_samples"), exist_ok=True)
        model.eval()

        ious, sample_data = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                for i in range(images.size(0)):
                    iou = compute_iou(outputs[i:i+1], labels[i:i+1])
                    # iou = self.compute_iou(outputs[i:i+1], labels[i:i+1])
                    ious.append(iou)
                    sample_data.append({
                        "input": images[i].cpu(),
                        "logits": outputs[i].cpu(),
                        "label": labels[i].cpu(),
                        "iou": iou
                    })

        # ---- choose representative samples by IoU ----
        n = len(ious)
        if n == 0:
            return
        ious = np.array(ious)
        sel_idx = []
        sample_percentiles = list(sample_percentiles) if sample_percentiles else []
        if sample_percentiles:
            sel_vals = np.percentile(ious, sample_percentiles)
            seen = set()
            for v in sel_vals:
                idx = int((np.abs(ious - v)).argmin())
                if idx not in seen:
                    sel_idx.append(idx)
                    seen.add(idx)
        else:
            sel_idx = list(range(n))

        for idx in sel_idx:
            d = sample_data[idx]
            save_path = os.path.join(
                root_dir,
                "test_samples",
                f"test_iou_{d['iou']:.3f}_{idx}.png",
            )
            visualize_sample_3d(
                d["input"],
                d["logits"],
                d["label"],
                save_path,
                miou=d["iou"],
                title_prefix="Test Sample",
            )


if __name__ == "__main__":
    tool = Torch3DSegmentationLearn()
    asyncio.run(tool.main())
