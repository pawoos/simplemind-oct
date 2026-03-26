"""
Tool Name: mask_smoothing
=================================

Description:
    Smooths the contours of binary masks (SMImage) in 2D or 3D using Gaussian filtering
    and morphological operations. Masks are assumed to contain only 0 and 1 values.

Parameters:
    - input_image (SMImage): Input binary mask to be processed (2D or 3D).
    - smoothing_method (str): One of "gaussian", "morphological", "combined".
        * "gaussian": Uses Gaussian blur followed by thresholding
        * "morphological": Uses opening followed by closing operations
        * "combined": Uses both Gaussian and morphological smoothing
    - smoothing_amount (float): Controls the amount of smoothing.
        * For gaussian: sigma value for Gaussian kernel (e.g., 1.0, 2.0)
        * For morphological: kernel size for structuring element (e.g., 3, 5)
        * For combined: applies to both operations
    - dimensionality (int, optional): 2 or 3 (default = 2).
        Determines whether to apply smoothing in 2D or 3D.

Output:
    - SMImage: The smoothed binary mask (values 0 and 1).

Example JSON Plan:
    # 2D Gaussian smoothing
    "mask_processing-smooth_gaussian_2d": {
        "code": "mask_smoothing.py",
        "context": "./tools/mask_processing/mask_smoothing/",
        "input_image": "from neural_net-torch_seg",
        "smoothing_method": "gaussian",
        "smoothing_amount": 1.5,
        "dimensionality": 2
    }

    # 3D morphological smoothing
    "mask_processing-smooth_morph_3d": {
        "code": "mask_smoothing.py",
        "context": "./tools/mask_processing/mask_smoothing/",
        "input_image": "from neural_net-torch_seg",
        "smoothing_method": "morphological",
        "smoothing_amount": 3,
        "dimensionality": 3
    }

    # Combined smoothing approach
    "mask_processing-smooth_combined": {
        "code": "mask_smoothing.py",
        "context": "./tools/mask_processing/mask_smoothing/",
        "input_image": "from neural_net-torch_seg",
        "smoothing_method": "combined",
        "smoothing_amount": 2.0,
        "dimensionality": 2
    }

Notes:
    - Input and output masks must contain only values 0 and 1.
    - Uses scipy.ndimage for Gaussian filtering and skimage.morphology for morphological operations.
    - Gaussian method: applies blur then thresholds at 0.5 to maintain binary nature.
    - Morphological method: uses opening to remove small protrusions, then closing to fill gaps.
    - Combined method: applies Gaussian smoothing first, then morphological refinement.
"""

import asyncio
import numpy as np
from scipy import ndimage
import skimage.morphology as morph

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage

class MaskSmoothing(SMSampleProcessor):

    async def setup(
        self,
        *,
        smoothing_method: str,
        smoothing_amount: float,
        dimensionality: int = 2  # 2 or 3
    ) -> None:

        self.smoothing_method = smoothing_method
        self.smoothing_amount = smoothing_amount
        self.dimensionality = dimensionality

        # Validate parameters
        attributes_verified_dict, attribute_error_msg_dict = self.verify_attributes(
            smoothing_method, smoothing_amount, dimensionality
        )

        params_ok = True
        for key in attributes_verified_dict.keys():
            if not attributes_verified_dict[key]:
                await self.log_message(f"{attribute_error_msg_dict[key]}")
                params_ok = False

        self.params_valid = params_ok

    async def execute(
        self,
        *,
        input_image: SMImage,
        smoothing_method: str,
    ) -> SMImage:

        if input_image is None:
            return None

        if not self.params_valid:
            self.print_error("Invalid smoothing parameters", warning=True)
            return None

        roi_array = input_image.pixel_array

        # Validate mask contains only 0 and 1
        unique_vals = np.unique(roi_array)
        if not np.all(np.isin(unique_vals, [0, 1])):
            self.print_error(f"Input mask contains non-binary values: {unique_vals}", warning=True)
            return None

        smoothed_mask = self.smooth_mask(roi_array)

        # Ensure output stays 0/1
        smoothed_mask = (smoothed_mask > 0.5).astype(np.uint8)

        return SMImage(input_image.metadata, smoothed_mask, input_image.label_array)

    def verify_attributes(self, smoothing_method, smoothing_amount, dimensionality):
        method_attributes = ["gaussian", "morphological", "combined"]

        attributes_verified_dict = {
            "method_verified": False,
            "amount_verified": False,
            "dim_verified": False,
        }
        attribute_error_msg_dict = {
            "method_verified": "Please assign 'gaussian', 'morphological', or 'combined' to 'smoothing_method'.",
            "amount_verified": "Please assign a positive value to 'smoothing_amount'.",
            "dim_verified": "Dimensionality must be 2 or 3.",
        }

        attributes_verified_dict["method_verified"] = smoothing_method in method_attributes
        attributes_verified_dict["amount_verified"] = smoothing_amount > 0
        attributes_verified_dict["dim_verified"] = dimensionality in [2, 3]

        return attributes_verified_dict, attribute_error_msg_dict

    def smooth_mask(self, roi_array):
        # Handle channel dimension from 4D SMImage ([C, Z, Y, X]) -> drop channel
        restore_channel = False
        if roi_array.ndim == 4:
            if roi_array.shape[0] != 1:
                raise ValueError(f"Expected single-channel mask, got shape {roi_array.shape}")
            roi_array = roi_array[0]  # now [Z, Y, X]
            restore_channel = True

        # Apply smoothing based on method
        if self.smoothing_method == "gaussian":
            smoothed = self.apply_gaussian_smoothing(roi_array)
        elif self.smoothing_method == "morphological":
            smoothed = self.apply_morphological_smoothing(roi_array)
        elif self.smoothing_method == "combined":
            # First apply Gaussian, then morphological
            gaussian_smoothed = self.apply_gaussian_smoothing(roi_array)
            smoothed = self.apply_morphological_smoothing(gaussian_smoothed)
        else:
            raise ValueError(f"Unsupported smoothing method: {self.smoothing_method}")

        # Restore channel dimension if needed
        if restore_channel:
            smoothed = np.expand_dims(smoothed, axis=0)

        # For 2D inputs originally shaped [1, H, W], maintain that shape
        if smoothed.ndim == 2:
            smoothed = np.expand_dims(smoothed, axis=0)

        return smoothed

    def apply_gaussian_smoothing(self, roi_array):
        """Apply Gaussian smoothing followed by thresholding"""
        # Convert to float for Gaussian filtering
        float_array = roi_array.astype(np.float32)
        
        # Apply Gaussian filter
        if self.dimensionality == 2 and roi_array.ndim == 3:
            # Apply 2D Gaussian to each slice
            smoothed_slices = []
            for z in range(roi_array.shape[0]):
                smoothed_slice = ndimage.gaussian_filter(float_array[z], sigma=self.smoothing_amount)
                smoothed_slices.append(smoothed_slice)
            smoothed = np.stack(smoothed_slices, axis=0)
        else:
            # Apply Gaussian filter directly
            smoothed = ndimage.gaussian_filter(float_array, sigma=self.smoothing_amount)
        
        return smoothed

    def apply_morphological_smoothing(self, roi_array):
        """Apply morphological opening followed by closing"""
        # Create structuring element
        kernel_size = int(self.smoothing_amount)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
        
        if self.dimensionality == 2:
            kernel = morph.disk(kernel_size // 2)
        else:
            kernel = morph.ball(kernel_size // 2)
        
        def apply_morph_ops(arr):
            # Opening removes small protrusions
            opened = morph.opening(arr, kernel)
            # Closing fills small gaps
            closed = morph.closing(opened, kernel)
            return closed
        
        # Apply morphological operations
        if self.dimensionality == 2 and roi_array.ndim == 3:
            # Apply 2D morphology to each slice
            smoothed_slices = []
            for z in range(roi_array.shape[0]):
                smoothed_slice = apply_morph_ops(roi_array[z])
                smoothed_slices.append(smoothed_slice)
            smoothed = np.stack(smoothed_slices, axis=0)
        else:
            smoothed = apply_morph_ops(roi_array)
        
        return smoothed.astype(np.float32)


if __name__ == "__main__":
    tool = MaskSmoothing()
    asyncio.run(tool.main())