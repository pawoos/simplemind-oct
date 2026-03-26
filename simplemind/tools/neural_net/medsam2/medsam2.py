"""
Tool Name: MedSAM2
=================================

Description:
    Apply MedSAM2 segmentation.

Parameters:            
    - input_image (SMImage): Input image to be segmented.
        Must be 3D.
    - prompt_mask (SMImage): Prompt mask.
        A bounding box is computed on this mask as a prompt for MedSAM2.
        The prompt must be 2D, operating on a single z-slice of the input image.
        If not, then None is output.

Output:
    - SMImage: The segmentation mask.
            
Example JSON Plan:
    "medsam2_mask": {
        "code": "medsam2.py",
        "chunk": "medsam2",
        "input_image": "from input_image",
        "prompt_mask": "from prompt_mask",
        "final_output": true
    },

Notes:
    - MedSAM2 library: https://github.com/bowang-lab/MedSAM2
"""

import asyncio
import numpy as np
import subprocess
from pathlib import Path
import sys

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID
from env_helper import setup_env, call_in_env # In smtool folder

# ENV_NAME = "medsam2"
# LIB_BASE_DIR = Path("../../../lib")
# LIB_DIR = Path("../../../lib") / "MedSAM2"

class MedSAM2(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        prompt_mask: SMImage,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None

        bb_dict = self.bounding_box_coords(prompt_mask.pixel_array, sample_id)
        if bb_dict is None:
            return None

        # prompt expects [x1, y1, x2, y2], so need to reverse the order of the tuples
        prompt = [*bb_dict['top_left'][::-1], *bb_dict['bottom_right'][::-1]]
        z_index = bb_dict['z']
        # self.print_log(f"z_index = {z_index}", sample_id)
        # self.print_log(f"prompt = {prompt}", sample_id)
        
        result_bytes = call_in_env(
            script_name="ms2.py",
            input_data=input_image.to_bytes(),
            script_args=["--z_index", str(z_index), "--prompt", str(prompt[0]), str(prompt[1]), str(prompt[2]), str(prompt[3])],
            env_name="medsam2",
            hash_dir=Path("../../../lib"),
            setup_env_func=setup_env
        )
        
        sm_image_result = SMImage.from_bytes(result_bytes)
        
        return sm_image_result


    def bounding_box_coords(self, arr: np.ndarray, sample_id: SMSampleID, axis: int = 0):
        """
        Returns the top-left and bottom-right coordinates of a nonzero region for a single slice mask.
        arr is required to be 3D
        Checks if nonzeros exist only on a single slice along `axis`.

        Parameters:
            arr (np.ndarray): Input 3D array.
            axis (int): Axis to interpret as the slicing axis (default=0).

        Returns:
            dict or None:
                - For 3D arrays with one nonzero slice:
                    {"z": slice_index, "top_left": (y_min, x_min), "bottom_right": (y_max, x_max)}
                - Returns None if:
                    * All values are zero, or
                    * Nonzeros exist in more than one slice (with printed warning)
        """
        if arr.ndim != 3:
            self.print_error(f"Expected a 3D array for the prompt mask, got {arr.ndim}D", sample_id)
            raise ValueError(f"Expected 3D array, got {arr.ndim}D")

        if axis < 0 or axis >= arr.ndim:
            raise ValueError(f"Axis {axis} out of bounds for array with shape {arr.shape}")

        # Find which slices have any nonzero elements
        nonzero_slices = np.any(arr != 0, axis=tuple(i for i in range(arr.ndim) if i != axis))
        slice_indices = np.flatnonzero(nonzero_slices)

        if len(slice_indices) == 0:
            return None  # all zeros

        if len(slice_indices) > 1:
            self.print_error(f"Prompt mask should be 2D, it spans multiple slices along axis {axis}, {slice_indices.tolist()}, returning None", sample_id, True)
            return None

        # Single slice — extract it
        z = int(slice_indices[0])
        arr_proj = np.take(arr, z, axis=axis)

        coords = np.argwhere(arr_proj)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return {"z": z, "top_left": (y_min, x_min), "bottom_right": (y_max, x_max)}


if __name__ == "__main__":   
    tool = MedSAM2()
    asyncio.run(tool.main())
