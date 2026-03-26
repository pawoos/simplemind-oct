#!/usr/bin/env python3
import os
import sys
import warnings

# ============================================================
# 🔧 CLEAN STDOUT SETUP
# ============================================================
# 1. Redirect warnings to stderr
def _warn_to_stderr(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = _warn_to_stderr
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'efficient_track_anything'")
warnings.filterwarnings("ignore")  # optionally silence all warnings

# 2. Disable known noisy framework logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"

# 3. Redirect all accidental stdout writes (from libraries) to stderr
class _StdoutRedirector:
    def write(self, data):
        # Skip empty writes or newlines-only
        if data.strip():
            sys.stderr.write(data)
    def flush(self):
        sys.stderr.flush()
    def isatty(self):
        return False  # needed for torch._dynamo, matplotlib, etc.

sys.stdout = _StdoutRedirector()

# ============================================================
# NORMAL IMPORTS
# ============================================================
import argparse
import numpy as np
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from PIL import Image
import matplotlib.pyplot as plt

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor_npz,
)

from sm_image import SMImage

# ============================================================
# CONFIG
# ============================================================
MEDSAM2_DIR = Path("../../../lib") / "MedSAM2"  # adjust path as needed


# ============================================================
# FUNCTIONS
# ============================================================
def preprocess_numpy(image_volume, box_z_idx, resize=True):
    original_slice = image_volume[box_z_idx]
    H, W = original_slice.shape
    
    if resize:
        img_pil = Image.fromarray(original_slice.astype(np.uint8))
        img_rgb = img_pil.convert("RGB").resize((512, 512))
        img_array = np.array(img_rgb).transpose(2, 0, 1) / 255.0  # (3, 512, 512)
        resized_hw = (512, 512)
    else:
        img_array = np.stack([original_slice]*3) / 255.0          # (3, H, W)
        resized_hw = (H, W)

    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img_array = (img_array - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).to(device)
    return img_tensor, (H, W), resized_hw


def _scale_box_to_resized(box, orig_hw, resized_hw):
    H, W = orig_hw
    h_res, w_res = resized_hw
    sx = w_res / float(W)
    sy = h_res / float(H)
    x0, y0, x1, y1 = box
    return [int(round(x0 * sx)), int(round(y0 * sy)),
            int(round(x1 * sx)), int(round(y1 * sy))]


def segment_with_prompt(input_array: np.ndarray, prompt_box, box_z_idx=0):
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()

    CONFIG_DIR = (MEDSAM2_DIR / "efficient_track_anything/configs").resolve()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(repo_root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_path = hf_hub_download(
            repo_id="wanglab/MedSAM2",
            filename="eff_medsam2_small_FLARE25_RECIST_baseline.pt",
            cache_dir=ckpt_dir,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Silence stdout during predictor initialization (patch noisy compilers)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            predictor = build_efficienttam_video_predictor_npz(
                "efficienttam_s_512x512.yaml",
                checkpoint_path,
                device=device,
            )

    img_tensor, orig_hw, resized_hw = preprocess_numpy(input_array, box_z_idx, resize=True)
    prompt_box_resized = _scale_box_to_resized(prompt_box, orig_hw, resized_hw)

    h_res, w_res = resized_hw
    with torch.no_grad():
        state = predictor.init_state(img_tensor, h_res, w_res)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=np.array(prompt_box_resized, dtype=np.int32),
        )
        mask = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)

    return mask


def main(input_array: np.ndarray, prompt, z_index):
    mask = segment_with_prompt(input_array, prompt, z_index)

    # --- Create full 3D mask consistent with input volume ---
    if mask.shape != input_array.shape:
        D, H, W = input_array.shape
        mask_3d = np.zeros((D, H, W), dtype=np.uint8)

        # Resize mask back to original H×W if needed
        if mask.shape != (H, W):
            mask_resized = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))
        else:
            mask_resized = mask

        mask_3d[z_index] = mask_resized
        mask = mask_3d
    
    return mask


# ============================================================
# MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a CT slice using MedSAM2 and a box prompt.")
    parser.add_argument("--prompt", nargs=4, type=int, required=True,
                        help="Bounding box: x0 y0 x1 y1 (in ORIGINAL image coords)")
    parser.add_argument("--z_index", type=int, default=0, help="Z-slice index (default: 0)")
    args = parser.parse_args()

    data = sys.stdin.buffer.read()
    input_image = SMImage.from_bytes(data)

    mask = main(input_image.pixel_array, args.prompt, args.z_index)
    sm_image_result = SMImage(None, mask)

    # Write serialized bytes to clean stdout
    sys.__stdout__.buffer.write(sm_image_result.to_bytes())
    sys.__stdout__.flush()
