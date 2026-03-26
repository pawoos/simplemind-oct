"""
Tool Name: torch_segmentation_learn
=================================

Description:
    Trains a deep convolutional neural network (CNN) for segmentation.
        Currently single-channel only, but there is no reason for this to be inherently the case (the internal libraries can support multi channel).

Parameters:            
    - input_image (SMImage): Input image to be used for training (that include the reference mask as label_data).
        These are passed through by the `execute` method and are aggregated into the `results` arg of the `aggregate` method, where training is performed.
    - model_backbone (str): Currently supported models are defined in models/unet_zoo.py.
        See __init__.py
    - pretrained (bool, optional): Apply an available pretrained model (typically "imagenet"). Default = false.
        If the pretrained model is RGB, the UNet wrapper adapts timm’s encoder weights internally (averages the pretrained RGB filters or otherwise adapts them to a single channel).
    - encoder_unfreeze_mode (str, optional): One of {"frozen", "gradual", "immediate"}.
        - "frozen": keep the pretrained encoder weights frozen throughout training.
        - "gradual": start frozen and progressively unfreeze encoder stages at fixed epoch percentages (deepest layers first).
        - "immediate": leave all encoder weights trainable from the start (default).
    - num_classes (int, optional): number of class labels to be included in the segmentation prediction (default 1).
    - optimizer (str, optional): "Adam" or "AdamW". Default = "Adam".
    - learning_rate (float): Training parameter.
    - lr_factor (float, optional): Reduce LR by this fraction after a number of stagnant epochs (specified by patience).
        Default = 0.5.
    - lr_patience (int, optional): Reduce LR by factor after this number of stagnant epochs. Default = 5.
    - label_smoothing (float, optional): Amount of softening applied to binary labels (range [0.0–1.0]).
        A small value (e.g., 0.05) replaces hard 0/1 labels with softened probabilities (1→0.975, 0→0.025) near uncertain regions.
        Acts as a regularizer that reduces overconfidence and improves generalization.
        Set to 0.0 to disable (default).
    - label_smooth_edge_width (float, optional): Width (in pixels) around the mask edge where label smoothing is applied.
        Pixels farther than this distance from the object boundary remain hard 0/1 labels.
        If set to 0.0, smoothing is applied uniformly across the mask.
        Typical range: 2 – 5 pixels for medical segmentation.
        Default = 3.0.
    - dropout (float, optional): Probability of randomly deactivating neurons during training to reduce overfitting. Default = 0.0 (disabled).
    - batch_size (int): Training parameter.
    - num_epochs (int): Training parameter.
    - early_stop_patience (int, optional): Stop after this number of epochs with no improvement. Default = 20.
    - gradual_unfreeze_percentages (list[float], optional):
        Epoch percentages (default [0.3, 0.6, 0.9]) that trigger stage-by-stage unfreezing when `encoder_unfreeze_mode="gradual"`.
        Each threshold unfroze the next group of encoder stages working from the deepest layers back toward the input (approximately same number of layers unfrozen at each stage), so the feature extractor is fine-tuned progressively.
    - gradient_accumulation (int, optional):
        Number of mini-batches to accumulate before each optimizer step (simulates a larger effective batch).
    - loss_dice_weight / loss_ce_weight (float, optional):
        Relative weights for the Dice vs CE components when `num_classes == 1`.
        For multi-class segmentation these weights are ignored.
    - test_sample_percentiles (list[float], optional):
        Percentiles used to pick representative test sample images.
        Default = [10, 25, 26, 49, 50, 51, 75, 76, 90] (yields ~9 samples).
    - test_iou_percentiles (list[float], optional):
        Percentiles used to log summary IoU stats in the test section.
        Default = [10, 25, 50, 75, 90].
    - split_ratio (list[float, float, float]): Training, validation, test split. Default = [6, 2, 2].
    - split_seed (int): Default = 42.
    - gpu_num (int, optional): GPU to be used for computation. Default = 0.
        If GPU not available, falls back to CPU.
    - output_dir (str): Default = "../output".

    # ---- Augmentation parameters ----
    - aug_h_flip_prob : float, optional
        Probability of applying horizontal flip. Default = 0.0.
    - v_flip_prob : float, optional
        Probability of applying vertical flip. Default = 0.0.
    - aug_degrees : float, optional
        Maximum rotation angle in degrees for random affine transformations. Default = 0.0.
    - aug_translate : list[float, float], optional
        Fractional translation in x and y (e.g., [0.05, 0.05] → up to 5% shift). Default = [0.0, 0.0].
    - aug_scale : list[float, float], optional
        Scaling range for random zoom (e.g., [0.95, 1.05] → ±5% scale). Default = [1.0, 1.0].
    - aug_persp_distortion_scale : float, optional
        Strength of perspective warp (0 = none). Typical range 0–0.5. Default = 0.0.
    - aug_persp_p : float, optional
        Probability of applying perspective transform. Default = 0.0.
    - aug_deformation_alpha : float, optional
        Elastic deformation strength (larger → more intense warping). Default = 0.0.
    - aug_deformation_sigma : float, optional
        Gaussian kernel width controlling smoothness of the deformation field. Default = 5.
    - aug_deformation_p : float, optional
        Probability of applying elastic deformation. Default = 0.0.

Output:
    - None (weights file and log file outputs are not directly accessible to other tools).
            
Example JSON Plan:
  "neural_net-torch_seg": {
    "code": "torch_segmentation.py",
    "code_learn": "torch_segmentation_learn.py",
    "input_image": "from image_preprocessing-clahe",
    "model_backbone": "resnet50",
    "weights_path": "/home/matt/weights/right_lung_weights.pth",
    "num_classes": 1,
    "map_output_dir": "from arg output_dir",
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 100
  }
  
Notes:
    - Uses the pytorch package: https://docs.pytorch.org/docs/stable/
    - Online resource for basic understanding of training parameters: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    - torch_segmentation_learn writes weights and log files, but no output to the Blackboard, i.e., no output that can be used as input to other tools
    - For binary segmentation, uses DiceBCELoss = BCEWithLogitsLoss + DiceLoss
        BCEWithLogitsLoss: Per-pixel probability accuracy
        DiceLoss: Mask overlap (F1/Dice coefficient) - better for imbalanced or small structures
        DiceBCELoss (combined): Balances pixel-accuracy and region overlap
        DiceLoss = 1 - Dice, so the goal is to minimize loss (0.0 is perfect)
    - If augmentation is applied, then approximately 5% of augmented samples are saved to a sample_augmentations folder.
"""

import asyncio
import os
import re
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from models import *
from sklearn.model_selection import train_test_split
from losses import DiceBCELoss, MultiClassDiceCELoss

from sm_sample_aggregator import SMSampleAggregator
from sm_image import SMImage
from sm_image_dataset import SMImageDataset

from train_utils import (
    compute_iou,
    plot_loss_miou,
    train_val_test_split,
    visualize_sample_2d,
)


def _batch_dice_loss(outputs: torch.Tensor, labels: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Mean Dice loss for a batch using sigmoid probabilities.
    """
    probs = torch.sigmoid(outputs)
    probs = probs.view(probs.size(0), -1)
    targets = labels.view(labels.size(0), -1).float()
    intersection = (probs * targets).sum(dim=1)
    dice = 1 - (2 * intersection + smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean()

class TorchSegmentationLearn(SMSampleAggregator):

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
        model_backbone: str,
        pretrained: bool = False,
        num_classes: int = 1,
        loss_dice_weight: float = 1.0,
        loss_ce_weight: float = 1.0,
        label_smoothing: float = 0.0,
        label_smooth_edge_width: float = 3.0,
        optimizer: str = "Adam",
        learning_rate: float,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        weight_decay: float = 0.0,
        encoder_unfreeze_mode: str = "immediate",
        gradual_unfreeze_percentages: list = [0.3, 0.6, 0.9],
        dropout: float = 0.0,        
        aug_h_flip_prob: float = 0.0,
        aug_v_flip_prob: float = 0.0,
        aug_degrees: float = 0.0,
        aug_translate: list = [0.0, 0.0], # Tuple[float, float]
        aug_scale: list = [1.0, 1.0], # Tuple[float, float]
        aug_persp_distortion_scale: float = 0.0, 
        aug_persp_p: float = 0.0,
        aug_deformation_alpha: float = 0.0, 
        aug_deformation_sigma: float = 5.0, 
        aug_deformation_p: float = 0.0,
        test_sample_percentiles: list = [10, 25, 26, 49, 50, 51, 75, 76, 90],
        test_iou_percentiles: list = [10, 25, 50, 75, 90],
        batch_size: int,
        num_epochs: int,
        early_stop_patience: int = 20,
        gradient_accumulation: int = 1,
        split_ratio: list = [6, 2, 2],
        split_seed: int = 42,
        gpu_num: int = 0,
        output_dir: str = None
    ) -> None:

        # Set up log file paths        
        out_dir = self.resolve_output_dir(output_dir, dataset_id)
        root_dir = self.dataset_output_path(out_dir, self.name().replace(f"-{self.plan_id}", ""))             
        log_training_file = os.path.join(root_dir, "log_training.txt")
        log_model_file = os.path.join(root_dir, "log_model.txt")
        plot_file = os.path.join(root_dir, "loss_miou_curve.png")
        
        if float(gpu_num) < 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # uses all available
        else:
            device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")

        import warnings
        import logging
        warnings.filterwarnings("ignore", message=".*Unexpected key.*")
        warnings.filterwarnings("ignore", message=".*verbose parameter is deprecated.*")
        logging.getLogger("timm").setLevel(logging.ERROR)

        model = unet_zoo.Model(
            backbone=model_backbone,    # backbone network name
            in_channels=1,              # input channels (1 for gray-scale images, 3 for RGB, etc.)
            num_classes=num_classes,    # output channels (number of classes in your dataset)
            pretrained=pretrained,
            encoder_freeze=(encoder_unfreeze_mode.lower() in {"frozen", "gradual"}),
        )
        TorchSegmentationLearn.add_decoder_dropout(model, p=dropout, log_fn=self.print_log)

        model = model.to(device)
        if float(gpu_num) < 0:
            model = torch.nn.DataParallel(model,)
        else:
            model = torch.nn.DataParallel(model,device_ids=[gpu_num])
        with open(log_model_file, 'w') as f:
            f.write(f"{model}")

        def resolve_module_path(root, module_path):
            module = root
            for token in module_path.split("."):
                if hasattr(module, token):
                    module = getattr(module, token)
                elif token.isdigit():
                    idx = int(token)
                    try:
                        module = module[idx]
                    except (TypeError, IndexError, KeyError):
                        module = module._modules.get(token)
                elif token in getattr(module, "_modules", {}):
                    module = module._modules[token]
                else:
                    return None
            return module

        def collect_encoder_stages(wrapped_model):
            encoder = wrapped_model.module.encoder if isinstance(wrapped_model, torch.nn.DataParallel) else wrapped_model.encoder
            modules = []
            for info in getattr(encoder, "feature_info", []):
                mod = resolve_module_path(encoder, info["module"])
                if mod is not None:
                    modules.append(mod)
            return modules

        def set_stage_trainable(modules, trainable):
            for module in modules:
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad = trainable

        def split_list(indices, n_chunks):
            if n_chunks <= 0:
                return []
            k = len(indices)
            base = k // n_chunks
            rem = k % n_chunks
            chunks = []
            start = 0
            for i in range(n_chunks):
                size = base + (1 if i < rem else 0)
                chunks.append(indices[start:start+size])
                start += size
            return chunks

        encoder_stage_modules = collect_encoder_stages(model)
        encoder_unfreeze_mode = (encoder_unfreeze_mode or "immediate").lower()
        valid_modes = {"frozen", "gradual", "immediate"}
        if encoder_unfreeze_mode not in valid_modes:
            raise ValueError(f"encoder_unfreeze_mode must be one of {valid_modes}, got '{encoder_unfreeze_mode}'")

        unfreeze_schedule = []
        unfreeze_pointer = 0

        if encoder_unfreeze_mode == "frozen":
            set_stage_trainable(encoder_stage_modules, False)
        elif encoder_unfreeze_mode == "gradual":
            set_stage_trainable(encoder_stage_modules, False)
            if encoder_stage_modules:
                percents = [p for p in gradual_unfreeze_percentages if 0 < p <= 1]
                if not percents:
                    percents = [0.3, 0.6, 0.9]
                percents = sorted(set(percents))
                stage_indices = list(range(len(encoder_stage_modules) - 1, -1, -1))
                chunks = split_list(stage_indices, len(percents))
                for percent, chunk in zip(percents, chunks):
                    if not chunk:
                        continue
                    epoch_threshold = max(1, min(num_epochs, int(round(percent * num_epochs))))
                    modules_chunk = [encoder_stage_modules[idx] for idx in chunk]
                    unfreeze_schedule.append((epoch_threshold, modules_chunk))

        is_binary = num_classes == 1
        if is_binary:
            criterion = None
            bce_loss_fn = nn.BCEWithLogitsLoss()
        else:
            criterion = MultiClassDiceCELoss()
            bce_loss_fn = None

        def compute_loss_components(outputs, labels):
            if is_binary:
                dice_tensor = _batch_dice_loss(outputs, labels)
                ce_tensor = bce_loss_fn(outputs, labels)
                loss = loss_dice_weight * dice_tensor + loss_ce_weight * ce_tensor
                return loss, dice_tensor.item(), ce_tensor.item()
            loss = criterion(outputs, labels)
            return loss, None, None

        # Define an optimizer
        if optimizer == "Adam":
            opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            err_string = f"optimizer selection ({opt}) is invalid: 'Adam' and 'AdamW' are allowed"
            self.print_error(err_string)
            raise ValueError(err_string)

        # Set up augmentation parameters
        augmentation_params = {
            'h_flip_prob': aug_h_flip_prob,
            'v_flip_prob': aug_v_flip_prob,
            'degrees': aug_degrees,
            'translate': tuple(aug_translate),
            'scale': tuple(aug_scale),
            'persp_distortion_scale': aug_persp_distortion_scale, 
            'persp_p': aug_persp_p,
            'deformation_alpha': aug_deformation_alpha, 
            'deformation_sigma': aug_deformation_sigma, 
            'deformation_p': aug_deformation_p,
            'vis_dir':root_dir
        }
        # Disable augmentation if all parameters correspond to a no-op configuration
        if TorchSegmentationLearn.is_no_augmentation_config(augmentation_params):
            augmentation_params = None
            self.print_log("Augmentation disabled (no-op configuration detected).")
        else:
            self.print_log("Augmentation enabled.")  
             
        # ==========================================================
        # PERFORM DATASET SPLIT
        # ==========================================================
        split_ratio_tuple = tuple(split_ratio) if not isinstance(split_ratio, tuple) else split_ratio
        train_list, val_list, test_list = train_val_test_split(results, split_ratio_tuple, split_seed)
        # train_split, val_split, test_split = split_ratio 
        # test_size_1 = (val_split+test_split)/(train_split+val_split+test_split)
        # test_size_2 = test_split/(val_split+test_split)
        # train_list, val_list = TorchSegmentationLearn.train_test_split(results, test_size=test_size_1, random_state=split_seed)
        # if test_split != 0:
        #     val_list, test_list = TorchSegmentationLearn.train_test_split(val_list, test_size=test_size_2, random_state=split_seed)

        train_dataset = SMImageDataset(train_list, 
                                        num_classes=num_classes, dim2=True,
                                        augmentation_params=augmentation_params, 
                                        label_smoothing=label_smoothing,
                                        label_smooth_edge_width=label_smooth_edge_width)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)        
        val_dataset = SMImageDataset(val_list, num_classes=num_classes, dim2=True, augmentation_params=None)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_dataset = SMImageDataset(test_list, num_classes=num_classes, dim2=True, augmentation_params=None)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        with open(log_training_file, 'w') as f:
            f.write(f"num training samples = {len(train_dataloader.dataset)}\n")        
            f.write(f"num validation samples = {len(val_dataloader.dataset)}\n") 
            f.write(f"num test samples = {len(test_dataloader.dataset)}\n") 
            f.write('-' * 50); f.flush()

        # --- Training setup ---
        best_miou = 0.0
        early_stop_counter = 0
        es_patience = early_stop_patience  # stop after this number of epochs with no improvement

        # Learning rate scheduler: reduce LR by 0.5 after 5 stagnant epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=lr_factor, patience=lr_patience, verbose=False
        )
        gradient_accumulation = max(1, int(gradient_accumulation))
        
        # ==========================================================
        # TRAINING START
        # ==========================================================        
        with open(log_training_file, 'a') as f:  # Open the log file
            for epoch in range(num_epochs):
                free, total = torch.cuda.mem_get_info()
                current_lr = opt.param_groups[0]['lr']
                f.write(f"Epoch {epoch+1}/{num_epochs}\n")
                f.write(f"Available: {free / 1024**2:.2f} MB / {total / 1024**2:.2f} MB\n")
                f.write(f"Learning rate: {current_lr:.6f}\n"); f.flush()
                
                if encoder_unfreeze_mode == "gradual":
                    while unfreeze_pointer < len(unfreeze_schedule) and epoch >= unfreeze_schedule[unfreeze_pointer][0]:
                        set_stage_trainable(unfreeze_schedule[unfreeze_pointer][1], True)
                        unfreeze_pointer += 1
                
                model.train()
                running_loss = 0.0
                running_dice = 0.0
                running_ce = 0.0
                opt.zero_grad()
                total_steps = len(train_dataloader)
                for step, (inputs, labels) in enumerate(train_dataloader, start=1):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss, dice_batch, ce_batch = compute_loss_components(outputs, labels)
                    (loss / gradient_accumulation).backward()
                    if step % gradient_accumulation == 0 or step == total_steps:
                        opt.step()
                        opt.zero_grad()

                    running_loss += loss.item() * inputs.size(0)
                    if dice_batch is not None:
                        running_dice += dice_batch * inputs.size(0)
                    if ce_batch is not None:
                        running_ce += ce_batch * inputs.size(0)

                epoch_loss = running_loss / len(train_dataloader.dataset)
                epoch_dice = running_dice / len(train_dataloader.dataset) if running_dice > 0 else 0.0
                epoch_ce = running_ce / len(train_dataloader.dataset) if running_ce > 0 else 0.0
                f.write(f"Train Loss: {epoch_loss:.4f}\n"); f.flush()
                if is_binary:
                    f.write(f"Train Dice Loss: {epoch_dice:.4f}\n")
                    f.write(f"Train CE Loss: {epoch_ce:.4f}\n")

                # Compute validation set metrics
                model.eval()
                running_val_loss = 0.0
                running_val_dice = 0.0
                running_val_ce = 0.0
                running_miou = 0.0
                with torch.no_grad():
                    for inputs, labels in val_dataloader:    
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)

                        val_loss, dice_val, ce_val = compute_loss_components(outputs, labels)
                        running_val_loss += val_loss.item() * inputs.size(0)
                        if dice_val is not None:
                            running_val_dice += dice_val * inputs.size(0)
                        if ce_val is not None:
                            running_val_ce += ce_val * inputs.size(0)
                        running_miou += compute_iou(outputs, labels) * inputs.size(0)
                        # running_miou += TorchSegmentationLearn.compute_iou(outputs, labels) * inputs.size(0)

                epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
                epoch_val_dice = running_val_dice / len(val_dataloader.dataset) if running_val_dice > 0 else 0.0
                epoch_val_ce = running_val_ce / len(val_dataloader.dataset) if running_val_ce > 0 else 0.0
                epoch_miou = running_miou / len(val_dataloader.dataset)
                f.write(f"Validation Loss: {epoch_val_loss:.4f}\n")
                if is_binary:
                    f.write(f"Validation Dice Loss: {epoch_val_dice:.4f}\n")
                    f.write(f"Validation CE Loss: {epoch_val_ce:.4f}\n")
                f.write(f"Validation mIoU: {epoch_miou:.4f}\n")
                if epoch % 5 == 0:
                    self.save_epoch_images(inputs, outputs, labels, num_classes, epoch, root_dir, miou=epoch_miou)

                # ---- SCHEDULER STEP ----
                scheduler.step(epoch_miou)

                # ---- CHECK FOR IMPROVEMENT ----
                if epoch_miou > best_miou:
                    best_miou = epoch_miou
                    early_stop_counter = 0
                    torch.save(model.state_dict(), os.path.join(root_dir, "checkpoint.pth"))
                    f.write(f"Improved! Saved best model (mIoU={best_miou:.4f})\n")
                else:
                    early_stop_counter += 1
                    f.write(f"No improvement ({early_stop_counter}/{es_patience})\n")

                # ---- EARLY STOPPING ----
                if early_stop_counter >= es_patience:
                    f.write(f"Early stopping after {es_patience} stagnant epochs.\n")
                    break

                f.write('-' * 50); f.flush()

        plot_loss_miou(log_training_file, plot_file) 
        # TorchSegmentationLearn.plot_loss_miou(log_training_file, plot_file) 

        # ==========================================================
        # TEST SET EVALUATION (after training)
        # ==========================================================
        self.print_log("Evaluating on test set...")
        best_model_path = os.path.join(root_dir, "checkpoint.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            self.print_log(f"Loaded best model: {best_model_path}")
        else:
            self.print_log("Warning: best model checkpoint not found; using last epoch model.")

        model.eval()

        test_loss_total = 0.0
        iou_scores = []
        sample_images = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                # ---- Compute loss ----
                loss, _, _ = compute_loss_components(outputs, labels)
                test_loss_total += loss.item() * inputs.size(0)

                # ---- Compute per-sample IoU ----
                probs = torch.sigmoid(outputs)

                for i in range(inputs.size(0)):
                    iou = compute_iou(outputs[i:i+1], labels[i:i+1])
                    # iou = TorchSegmentationLearn.compute_iou(outputs[i:i+1], labels[i:i+1])
                    iou_scores.append(iou)

                    sample_images.append({
                        "input": inputs[i].detach().cpu().numpy(),
                        "prob": probs[i].detach().cpu().numpy(),
                        "label": labels[i].detach().cpu().numpy(),
                        "iou": iou
                    })

        # ---- Compute aggregate metrics ----
        test_loss = test_loss_total / len(test_dataloader.dataset)
        test_miou = sum(iou_scores) / len(iou_scores)
        iou_array = np.array(iou_scores)

        def percentile_nearest(arr, perc):
            try:
                return np.percentile(arr, perc, method="nearest")
            except TypeError:
                return np.percentile(arr, perc, interpolation="nearest")

        with open(log_training_file, 'a') as f:
            f.write('-' * 50 + "\n")
            f.write(f"Final Test Loss: {test_loss:.4f}\n")
            f.write(f"Final Test mIoU: {test_miou:.4f}\n")
            log_percentiles = list(test_iou_percentiles) if test_iou_percentiles else []
            if log_percentiles:
                log_vals = percentile_nearest(iou_array, log_percentiles) if len(iou_array) else []
                f.write("Test IoU percentiles:\n")
                for p, val in zip(log_percentiles, np.atleast_1d(log_vals)):
                    f.write(f"  p{p}: {val:.4f}\n")
            else:
                f.write("Test IoU percentiles: not configured; logging all IoUs\n")
                for idx, val in enumerate(iou_array):
                    f.write(f"  sample_{idx}: {val:.4f}\n")
            f.flush()

        # ==========================================================
        # SELECT REPRESENTATIVE TEST SAMPLES BY IoU
        # ==========================================================

        n = len(iou_array)

        # ---- Select images for visualization based on configured percentiles ----
        selection = []
        sample_percentiles = list(test_sample_percentiles) if test_sample_percentiles else []
        if sample_percentiles:
            sample_vals = np.percentile(iou_array, sample_percentiles)
            seen_indices = set()
            for sval in sample_vals:
                idx = int((np.abs(iou_array - sval)).argmin())
                if idx not in seen_indices:
                    selection.append(idx)
                    seen_indices.add(idx)
        else:
            selection = list(range(n))

        # ---- Save representative test images ----
        test_vis_dir = os.path.join(root_dir, "test_samples")
        os.makedirs(test_vis_dir, exist_ok=True)
        for idx in selection:
            data = sample_images[idx]
            img = np.squeeze(data["input"])
            pred = np.squeeze(data["prob"])
            label = np.squeeze(data["label"])
            save_path = os.path.join(
                test_vis_dir,
                f"test_iou_{data['iou']:.3f}_{idx}.png",
            )
            visualize_sample_2d(
                img,
                pred,
                label,
                save_path,
                miou=data["iou"],
                title_prefix="Test Sample",
            )

        return None

    def save_epoch_images(self, inputs, outputs, labels, num_classes, epoch, root_dir, miou=None):
        """
        Save the images from the inputs, outputs, and labels tensors.
        """
        epoch_dir = os.path.join(root_dir, "epoch_images")
        os.makedirs(epoch_dir, exist_ok=True)

        inputs_cpu = inputs.detach().cpu().numpy()
        probs_cpu = torch.sigmoid(outputs.detach().cpu()).numpy()
        labels_cpu = labels.detach().cpu().numpy()

        max_samples = min(inputs_cpu.shape[0], 3)
        for idx in range(max_samples):
            img = np.squeeze(inputs_cpu[idx])
            pred = np.squeeze(probs_cpu[idx])
            label = np.squeeze(labels_cpu[idx])
            save_path = os.path.join(epoch_dir, f"epoch_{epoch}_sample_{idx}.png")
            visualize_sample_2d(
                img,
                pred,
                label,
                save_path,
                miou=miou,
                title_prefix=f"Epoch {epoch} | Sample {idx}",
            )
        

    @staticmethod
    def add_decoder_dropout(model, p=0.1, log_fn=print):
        """
        Adds spatial dropout (Dropout2d) layers to the decoder blocks of a UNet-style model.

        This function inspects the model for a `decoder` attribute with a `blocks` list
        (as used in most UNet implementations, e.g., EfficientNet-UNet, ResNet-UNet, etc.).
        If found, it appends a `nn.Dropout2d(p)` layer to each decoder block, which helps
        regularize the model by randomly zeroing entire feature maps during training.

        Parameters
        ----------
        model : torch.nn.Module
            The UNet model to which dropout should be added.
            Must have `model.decoder.blocks` defined (a ModuleList of decoder stages).
        p : float, optional
            Dropout probability (default = 0.1). Set to 0.0 to disable dropout.
            Typical range: 0.1–0.3 for moderate regularization.
        log_fn : callable, optional
            Logging function for status messages (default = built-in `print`).
            Can be replaced with a custom logger (e.g., `self.print_log`).

        Notes
        -----
        - Uses `nn.Dropout2d`, which is designed for convolutional feature maps.
        It drops entire 2D feature channels instead of individual pixels.
        - This method is model-safe: if the decoder is not found, it logs a message
        and exits without raising an error.
        - Adding dropout layers increases stochasticity during training, which helps
        reduce overfitting, especially for small medical image datasets.

        Example
        -------
        >>> model = unet_zoo.Model(backbone='efficientnet_b0', num_classes=1, pretrained=True)
        >>> add_decoder_dropout(model, p=0.1, log_fn=self.print_log)
        Added Dropout2d(p=0.1) to 5 decoder blocks.
        """
        # Early exit if dropout probability is 0.0
        if p == 0.0:
            log_fn("Dropout disabled (p=0.0). No layers added.")
            return

        # Ensure model has a decoder with multiple convolutional blocks
        if hasattr(model, "decoder") and hasattr(model.decoder, "blocks"):
            num_blocks = len(model.decoder.blocks)

            # Append dropout to each decoder block
            for i, block in enumerate(model.decoder.blocks):
                block.add_module("dropout", nn.Dropout2d(p=p))

            log_fn(f"Added Dropout2d(p={p}) to {num_blocks} decoder blocks.")
        else:
            # Graceful fallback if model structure doesn't match UNet
            log_fn("Decoder structure not found — no dropout added.")

    @staticmethod
    def is_no_augmentation_config(aug: dict | None) -> bool:
        """Return True if augmentation settings correspond to 'no augmentation'."""
        if aug is None:
            return True

        def as_tuple(x):
            return tuple(x) if isinstance(x, (list, tuple)) else x

        return (
            aug.get('h_flip_prob', 0.0) == 0.0 and
            aug.get('v_flip_prob', 0.0) == 0.0 and
            aug.get('degrees', 0.0) == 0.0 and
            as_tuple(aug.get('translate', (0.0, 0.0))) == (0.0, 0.0) and
            as_tuple(aug.get('scale', (1.0, 1.0))) == (1.0, 1.0) and
            aug.get('persp_distortion_scale', 0.0) == 0.0 and
            aug.get('persp_p', 0.0) == 0.0 and
            (aug.get('deformation_alpha', 0.0) == 0.0 or aug.get('deformation_p', 0.0) == 0.0)
        )
    
if __name__ == "__main__":   
    tool = TorchSegmentationLearn()
    asyncio.run(tool.main())
