import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import sys

# Optional MONAI for 3D augmentation
try:
    from monai.transforms import Compose, RandFlipd, RandAffined, Rand3DElasticd
    _HAS_MONAI = True
except ImportError:
    _HAS_MONAI = False


class SMImageDataset(Dataset):
    """
    Dataset for SMImage objects with optional on-the-fly PyTorch augmentations.
    Supports both binary and multi-class segmentation.

    Features:
      - Automatic 2D/3D detection
      - Torchvision augmentations for 2D
      - MONAI augmentations for 3D (if installed)
      - Optional elastic deformation
      - Optional edge-aware label smoothing
      - Visualization of original vs augmented samples
        (for 3D: slices through label centroid)

    Parameters
    ----------
    smimage_list : list[SMImage]
        List of input SMImage objects (each containing pixel_array and label_array)
    num_classes : int
        Number of segmentation classes (1 for binary)
    dim2: bool = False
    augmentation_params : dict or None, optional
        Dictionary controlling the spatial augmentation pipeline. If None,
        no augmentations are applied. Expected keys:
        - **h_flip_prob** (float): Probability of horizontal flip (range [0.0–1.0]).
        - **degrees** (float): Maximum rotation angle in degrees for random affine transforms.
        - **translate** (tuple[float, float]): Max fractional translation in x and y
          directions (e.g., (0.05, 0.05) → up to 5% shift).
        - **scale** (tuple[float, float]): Scaling range for random affine transformations
          (e.g., (0.95, 1.05) → ±5% zoom).
        - **persp_distortion_scale** (float): Degree of perspective distortion (typical range 0–0.5).
        - **persp_p** (float): Probability of applying perspective transform.
        - **deformation_alpha** (float): Elastic deformation intensity (larger → more warp).
        - **deformation_sigma** (float): Gaussian smoothing for the deformation field (controls smoothness).
        - **deformation_p** (float): Probability of applying elastic deformation.
        - **vis_dir** (str or None): Optional output directory for saving augmentation visualizations for debugging.
            If augmentation parameters are provided and this directory parameter is not None then approx 5% of samples are visualized.
    """

    def __init__(self, smimage_list, num_classes=1, dim2=False, augmentation_params=None,
                 label_smoothing=0.0, label_smooth_edge_width=0.0):

        self.smimage_list = smimage_list
        self.num_classes = num_classes
        self.aug_params = augmentation_params or {}
        # print(f"self.aug_params: {self.aug_params}", file=sys.stderr)
        self.label_smoothing = label_smoothing
        self.label_smooth_edge_width = label_smooth_edge_width

        # Detect dimensionality
        shape = smimage_list[0].pixel_array.shape  # [C, Z, Y, X]
        if len(shape) != 4:
            raise ValueError(f"Expected [C, Z, Y, X], got {shape}")
        self._is_3d = shape[1] > 1

        if not self._is_3d:
            for smimage in smimage_list:
                smimage.pixel_array = smimage.pixel_array[:, 0, :, :] # [C, Z, Y, X] -> [C, Y, X]
                if smimage.label_array is not None:
                    smimage.label_array = smimage.label_array[:, 0, :, :] # [C, Z, Y, X] -> [C, Y, X]

        # Build augmentations
        if self._is_3d:
            self._build_3d_pipeline()
        else:
            self._build_2d_pipeline()


    # ---------------------------------------------------------------------
    # 2D augmentations (torchvision)
    # ---------------------------------------------------------------------
    def _build_2d_pipeline(self):
        p = self.aug_params
        if not p:
            self.geom_transform = None
            return

        self.geom_transform = T.Compose([
            T.RandomHorizontalFlip(p=p.get('h_flip_prob', 0.5)),
            T.RandomVerticalFlip(p=p.get('v_flip_prob', 0.0)),
            T.RandomAffine(
                degrees=p.get('degrees', 10),
                translate=tuple(p.get('translate', (0.05, 0.05))),
                scale=tuple(p.get('scale', (0.95, 1.05))),
                interpolation=InterpolationMode.NEAREST
            ),
            T.RandomPerspective(
                distortion_scale=p.get('persp_distortion_scale', 0.2),
                p=p.get('persp_p', 0.3),
                interpolation=InterpolationMode.NEAREST
            ),
        ])


    # ---------------------------------------------------------------------
    # 3D augmentations (MONAI)
    # ---------------------------------------------------------------------
    def _build_3d_pipeline(self):
        if not _HAS_MONAI:
            print("[SMImageDataset] MONAI not installed — 3D augmentations disabled.", file=sys.stderr)
            self.geom_transform_3d = None
            return

        p = self.aug_params
        if not p:
            self.geom_transform_3d = None
            return
        # print(f"p: {p}", file=sys.stderr)

        self.geom_transform_3d = Compose([
            RandFlipd(keys=["image","label"],
                      prob=p.get('h_flip_prob', 0.0), spatial_axis=2),
            RandFlipd(keys=["image","label"],
                      prob=p.get('v_flip_prob', 0.0), spatial_axis=1),
            RandAffined(
                keys=["image","label"],
                prob=p.get('affine_p', 0.0),
                rotate_range=p.get('rotate_range', (0, 0, 0)),
                scale_range=p.get('scale_range', (0, 0, 0)),
                mode=("bilinear","nearest")
            ),
            Rand3DElasticd(
                keys=["image","label"],
                prob=p.get('deformation_p', 0.0),
                sigma_range=p.get('deformation_sigma_range', (0,0)),
                magnitude_range=p.get('deformation_alpha_range', (0,0)),
                mode=("bilinear","nearest")
            )
        ])


    # ---------------------------------------------------------------------
    # Main dataset interface
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.smimage_list)

    def __getitem__(self, idx):
        smimage = self.smimage_list[idx]
        image = torch.tensor(smimage.pixel_array, dtype=torch.float32)
        label = torch.tensor(smimage.label_array,
                             dtype=torch.float32 if self.num_classes == 1 else torch.long)

        # Normalize binary masks to {0,1} in case they are stored as {0,255}, {0,2}, etc.
        if self.num_classes == 1:
            if label.max() > 1.0 or label.min() < 0.0:
                label = (label > 0).float()
        else:
            # For multi-class, enforce labels stay within [0, num_classes-1]
            lbl_min, lbl_max = label.min().item(), label.max().item()
            if lbl_min < 0 or lbl_max >= self.num_classes:
                raise ValueError(f"Label values out of range [0, {self.num_classes-1}]: min={lbl_min}, max={lbl_max}")

        orig_image = image.clone()
        orig_label = label.clone()

        # ---- 2D path ----
        if not self._is_3d:
            if self.geom_transform is not None:
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                image = self.geom_transform(image)
                torch.manual_seed(seed)
                label = self.geom_transform(label)

            # Elastic deformation (2D)
            if self.aug_params.get('deformation_p', 0.0) > 0:
                image, label = SMImageDataset.elastic_deformation(
                    image, label,
                    alpha=self.aug_params.get('deformation_alpha', 10),
                    sigma=self.aug_params.get('deformation_sigma', 5),
                    p=self.aug_params.get('deformation_p', 0.2)
                )

        # ---- 3D path ----
        else:
            if self.geom_transform_3d is not None:
                data = {"image": image, "label": label}
                data = self.geom_transform_3d(data)
                image, label = data["image"], data["label"]

        # Ensure consistent shape
        if not self._is_3d and image.ndim == 2:
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
        elif self._is_3d and image.ndim == 3:
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)

        # ---- Label smoothing ----
        if self.label_smoothing > 0 and self.num_classes == 1:
            label = self._apply_label_smoothing(label)

        # ---- Optional visualization ----
        vis_dir = self.aug_params.get("vis_dir")
        # print(f"vis_dir = {vis_dir}", file=sys.stderr)
        if vis_dir and torch.rand(1).item() < 0.05:  # 5% random visualization
            # print("calling _visualize_augmentation", file=sys.stderr)
            os.makedirs(os.path.join(vis_dir, "sample_augmentations"), exist_ok=True)
            self._visualize_augmentation(
                original_image=orig_image.cpu().numpy(),
                augmented_image=image.cpu().numpy(),
                original_mask=orig_label.cpu().numpy(),
                augmented_mask=label.cpu().numpy(),
                out_dir=os.path.join(vis_dir, "sample_augmentations"),
                prefix=f"sample_{idx}"
            )

        return image, label


    # ---------------------------------------------------------------------
    # Edge-aware label smoothing (works for 2D and 3D)
    # ---------------------------------------------------------------------
    def _apply_label_smoothing(self, label):
        eps = self.label_smoothing
        edge_width = self.label_smooth_edge_width or 0.0
        label_np = label.squeeze().cpu().numpy().astype(np.float32)

        from scipy.ndimage import distance_transform_edt
        label_bin = (label_np > 0.5).astype(np.uint8)
        dist_to_fg = distance_transform_edt(label_bin == 0)
        dist_to_bg = distance_transform_edt(label_bin == 1)
        boundary_band = np.maximum(dist_to_fg, dist_to_bg)

        if edge_width > 0:
            smoothing_factor = np.clip(1 - boundary_band / edge_width, 0, 1)
            soft_label = label_np * (1 - eps * smoothing_factor) + 0.5 * eps * smoothing_factor
        else:
            soft_label = label_np * (1 - eps) + 0.5 * eps

        soft_label[soft_label < 0.05] = 0.0
        soft_label = np.clip(soft_label, 0, 1)
        return torch.tensor(soft_label, dtype=torch.float32).unsqueeze(0)


    # ---------------------------------------------------------------------
    # 2D elastic deformation (unchanged)
    # ---------------------------------------------------------------------
    @staticmethod
    def elastic_deformation(image, mask, alpha=10, sigma=5, p=0.2):
        """
        Apply elastic deformation to both image and mask using a random
        displacement field smoothed by a Gaussian kernel (pure PyTorch).
        Supports [C,H,W] or [B,C,H,W] input (batch or single image).

        Parameters
        ----------
        image : torch.Tensor [C,H,W]
            Image tensor (float32, usually 1 channel for X-rays)
        mask : torch.Tensor [C,H,W]
            Label tensor (float32 or long)
        alpha : float
            Displacement intensity
        sigma : float
            Gaussian blur std dev controlling smoothness
        p : float
            Probability of applying the deformation

        Returns
        -------
        image, mask : torch.Tensor
            Deformed tensors of same shape
        """

        if torch.rand(1).item() > p:
            return image, mask
        if image.ndim == 4: image = image.squeeze(0)
        if mask.ndim == 4: mask = mask.squeeze(0)
        device = image.device
        _, H, W = image.shape
        dx = torch.randn(1, 1, H, W, device=device)
        dy = torch.randn(1, 1, H, W, device=device)
        size = int(2 * sigma + 1)
        grid = torch.arange(size, device=device) - sigma
        kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
        kernel = kernel / kernel.sum()
        gauss = kernel[:, None] * kernel[None, :]
        gauss = gauss.expand(1, 1, -1, -1)
        blur = torch.nn.Conv2d(1, 1, size, padding=size // 2, bias=False)
        blur.weight.data = gauss
        blur.weight.requires_grad = False
        dx = blur(dx) * alpha
        dy = blur(dy) * alpha
        xx, yy = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing="xy"
        )
        xx = xx.unsqueeze(0).float() + dx.squeeze(0)
        yy = yy.unsqueeze(0).float() + dy.squeeze(0)
        xx = 2 * (xx / (W - 1)) - 1
        yy = 2 * (yy / (H - 1)) - 1
        grid = torch.stack((xx, yy), dim=-1)
        image = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                              padding_mode="reflection", align_corners=True).squeeze(0)
        mask = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest",
                             padding_mode="reflection", align_corners=True).squeeze(0)
        return image, mask


    # ---------------------------------------------------------------------
    # Visualization of 2D and 3D augmentations (with red/yellow mask overlay)
    # ---------------------------------------------------------------------
    def _visualize_augmentation(self, original_image, augmented_image,
                                original_mask=None, augmented_mask=None,
                                out_dir="debug_augmentations", prefix="aug"):
        """
        Save comparison of original and augmented images with and without mask overlays.
        If the augmented mask is continuous (soft labels), the overlay uses a heatmap
        to visualize soft probabilities instead of a binary red overlay.

        Parameters
        ----------
        original_image : np.ndarray or torch.Tensor
            Original image, shape [H,W] or [1,H,W].
        augmented_image : np.ndarray or torch.Tensor
            Augmented image, shape [H,W] or [1,H,W].
        original_mask : np.ndarray or torch.Tensor, optional
            Original label mask, shape [H,W] or [1,H,W].
        augmented_mask : np.ndarray or torch.Tensor, optional
            Augmented (possibly softened) label mask, shape [H,W] or [1,H,W].
        out_dir : str
            Directory where PNGs are written.
        prefix : str
            Filename prefix.
        """
        os.makedirs(out_dir, exist_ok=True)

        def norm(x):
            x = np.array(x, dtype=np.float32)
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            x -= x.min()
            if x.max() > 0:
                x /= (x.max() + 1e-8)
            return x

        orig = norm(original_image)
        aug = norm(augmented_image)

        # --- Helper for soft mask overlay (red for 1.0, yellow for soft edges) ---
        def overlay_soft_mask(img, mask, alpha=0.6):
            if mask is None:
                return np.stack([img]*3, axis=-1)
            img_rgb = np.stack([img]*3, axis=-1)
            val = np.clip(mask, 0, 1)
            rgb = np.zeros((*val.shape, 3), dtype=np.float32)
            red_mask = val >= 1.0 - 1e-6
            yellow_mask = (val > 0) & (val < 1.0 - 1e-6)
            rgb[..., 0][red_mask | yellow_mask] = 1.0
            rgb[..., 1][yellow_mask] = 1.0
            overlay = img_rgb * (1 - alpha) + rgb * alpha
            return np.clip(overlay, 0, 1)

        # --- 2D visualization ---
        if not self._is_3d:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(orig, cmap="gray")
            axs[0].set_title("Original")
            axs[1].imshow(overlay_soft_mask(orig, original_mask.squeeze() if original_mask is not None else None))
            axs[1].set_title("Original + Mask")
            axs[2].imshow(aug, cmap="gray")
            axs[2].set_title("Augmented")
            axs[3].imshow(overlay_soft_mask(aug, augmented_mask.squeeze() if augmented_mask is not None else None))
            axs[3].set_title("Augmented + Mask")
            for ax in axs: ax.axis("off")

        # --- 3D visualization (centroid slices) ---
        else:
            aug = aug.squeeze()
            mask = augmented_mask.squeeze()
            if mask.sum() == 0:
                cz, cy, cx = np.array(mask.shape) // 2
            else:
                coords = np.argwhere(mask > 0.5)
                cz, cy, cx = coords.mean(axis=0).astype(int)

            planes = [
                (aug[cz, :, :], mask[cz, :, :], "Axial (Z)"),
                (aug[:, cy, :], mask[:, cy, :], "Coronal (Y)"),
                (aug[:, :, cx], mask[:, :, cx], "Sagittal (X)"),
            ]
            planes_orig = [
                (orig[0, cz, :, :], original_mask.squeeze()[cz, :, :], "Axial (Z)"),
                (orig[0, :, cy, :], original_mask.squeeze()[:, cy, :], "Coronal (Y)"),
                (orig[0, :, :, cx], original_mask.squeeze()[:, :, cx], "Sagittal (X)"),
            ]

            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            for i, (img, msk, title) in enumerate(planes_orig):
                axs[0, i].imshow(overlay_soft_mask(img, msk))
                axs[0, i].set_title(f"Original {title}")
                axs[0, i].axis("off")
            for i, (img, msk, title) in enumerate(planes):
                axs[1, i].imshow(overlay_soft_mask(img, msk))
                axs[1, i].set_title(f"Augmented {title}")
                axs[1, i].axis("off")
            # for i, (img, msk, title) in enumerate(planes):
            #     axs[0, i].imshow(overlay_soft_mask(img, msk))
            #     axs[0, i].set_title(f"Augmented {title}")
            #     axs[0, i].axis("off")
            # for i, (img, msk, title) in enumerate(planes_orig):
            #     axs[1, i].imshow(overlay_soft_mask(img, msk))
            #     axs[1, i].set_title(f"Original {title}")
            #     axs[1, i].axis("off")
                
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{prefix}_{os.getpid()}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
