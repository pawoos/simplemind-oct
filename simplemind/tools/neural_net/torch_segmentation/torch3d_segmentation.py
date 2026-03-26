"""
Tool Name: torch3d_segmentation
================================

Description:
    Runs 3D segmentation inference using a MONAI-based 3D UNet.

Parameters:            
    - input_image (SMImage): Input volumetric image for prediction (shape [C,D,H,W]).
    - weights_path (str): Path to the trained MONAI model weights (.pth).
    - prediction_threshold (float, optional): Threshold to binarize output (default = 0.5).
    - num_classes (int): Number of output channels/classes (default = 1).
    - gpu_num: GPU index (default = 0; use -1 for CPU).
    - model_channels (tuple[int], optional): Encoder/decoder feature channels.
    - strides (tuple[int], optional): Downsampling strides between levels.
    - map_output_dir (str, optional): Directory to write PNG previews of axial/coronal/sagittal overlays.

Output:
    - SMImage: Binary segmentation mask (3D volume).

Example JSON Plan:
  "neural_net-torch3d_seg": {
    "code": "torch3d_segmentation.py",
    "input_image": "from image_preprocessing-norm",
    "weights_path": "/home/matt/weights/kidney_model_3d.pth",
    "num_classes": 1,
    "map_output_dir": "from arg output_dir",
    "prediction_threshold": 0.5,
    "gpu_num": 0
  }
"""

import os, asyncio
import torch
import warnings
import logging
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from monai.networks.nets import UNet as UNet3D
from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from train_utils import visualize_sample_3d
from sm_sample_id import SMSampleID

# Suppress benign warnings about unexpected keys when loading adapted pretrained weights.
warnings.filterwarnings("ignore", message=".*Unexpected key.*")
warnings.filterwarnings("ignore", message=".*Unexpected keys.*")
logging.getLogger("timm").setLevel(logging.ERROR)

class Torch3DSegmentationDebug(SMSampleProcessor):

    async def setup(
        self, 
        *, 
        num_classes: int = 1,
        weights_path: str,
        gpu_num: int = 0,
        model_channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)

    ) -> None:

        if float(gpu_num) < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
             
        # ---------- model (load existing weights) ----------
        self.model = UNet3D(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=model_channels,
            strides=strides,
            num_res_units=2,
        ).to(self.device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Unexpected key.*")
            warnings.filterwarnings("ignore", message=".*Unexpected keys.*")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()   

    async def execute(
        self,
        *,
        input_image: SMImage,
        prediction_threshold: float = 0.5,
        model_specific_output_class: int = -1,
        map_output_dir: str = None,
        sample_id: SMSampleID
    ) -> SMImage:       
        if input_image is None:
            return None

        # --- Prepare input tensor ---
        image_array = input_image.pixel_array
        image_tensor = torch.from_numpy(image_array).float().to(self.device)
        # label_tensor = torch.from_numpy(input_image.label_array).float().to(self.device)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        # labels = label_tensor.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            pred_logits = self.model(image_tensor)

        if model_specific_output_class >= 0:
            pred_logits = pred_logits[:, model_specific_output_class:model_specific_output_class+1]

        prob_volume = torch.sigmoid(pred_logits[0]).detach().cpu().numpy()
        channel_prob = prob_volume[0] if prob_volume.ndim == 4 else prob_volume
        pred_mask = (channel_prob > prediction_threshold).astype(np.uint8)
        mask_with_channel = pred_mask[None, ...]
        sm_output = SMImage(input_image.metadata, mask_with_channel, input_image.label_array)
        
        # --- Optional PNG map ---
        if map_output_dir is not None:
            base_output = self.resolve_output_dir(map_output_dir, sample_id.dataset)
            os.makedirs(base_output, exist_ok=True)

            sample_dir = self.sample_output_path(base_output, sample_id)
            os.makedirs(sample_dir, exist_ok=True)

            # Save probability map PNG (slice grid)
            map_path = os.path.join(sample_dir, "prediction_map.png")
            self.save_prediction_map(image_array[0], channel_prob, map_path)

            # Save visualize_sample_3d overlay per case
            viz_path = os.path.join(sample_dir, "prediction_planes.png")
            input_tensor = torch.from_numpy(input_image.pixel_array)
            pred_tensor = pred_logits.detach().cpu()[0]
            label_tensor = torch.from_numpy(mask_with_channel)
            visualize_sample_3d(
                input_tensor,
                pred_tensor,
                label_tensor,
                viz_path,
                miou=None,
                title_prefix="Inference",
            )        
        
        return sm_output

    @staticmethod
    def save_prediction_map(image_volume, prob_volume, save_path):
        """Save a grid of slices showing the probability map."""

        percents = np.linspace(0, 1, 7, endpoint=False)
        slices = np.floor(percents * image_volume.shape[0]).astype(int)
        slices = np.unique(slices)

        fig, axes = plt.subplots(2, len(slices), figsize=(len(slices)*3, 6))
        for idx, z in enumerate(slices):
            axes[0, idx].imshow(image_volume[z], cmap="gray")
            axes[0, idx].axis("off")
            axes[0, idx].set_title(f"{(z / image_volume.shape[0]) * 100:.0f}%")

            axes[1, idx].imshow(prob_volume[z], cmap="gray")
            axes[1, idx].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    tool = Torch3DSegmentationDebug()
    asyncio.run(tool.main())
