"""
Tool Name: half_and_flip
=================================

Description:
    Splits an image (or mask) in half vertically and flips the right half horizontally. 
    Outputs both halves separately or combined.

Parameters:            
    - input_image (SMImage): Input image to be split and flipped.
    - side (str, optional): Which side to return - 'left', 'right', or 'both'. Default is 'both'.
    - save_png (bool, optional): Whether to save output as PNG file. Default is False.
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - SMImage: The processed image (left half, right half flipped, or both stacked vertically).
            
Example JSON Plan:
    "image_processing-half_and_flip": {
        "code": "half_and_flip.py",
        "input_image": "from input_image",
        "side": "both",
        "save_png": false,
        "debug": false
    }

Notes:
    - The right half is flipped horizontally before output.
    - When side='both', both halves are stacked vertically for visualization.
    - Supports both 2D and 3D images with proper metadata handling.
"""

import asyncio
import os
import numpy as np
import cv2

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID


class HalfAndFlip(SMSampleProcessor):
    async def execute(
        self,
        *,
        input_image: SMImage,
        side: str = "both",
        save_png: bool = False,
        debug: bool = False,
        sample_id: SMSampleID
    ) -> SMImage:
        if input_image is None:
            return None

        # Extract input
        input_array = input_image.pixel_array

        if debug:
            self.print_log(f"Input shape: {input_array.shape}", sample_id)

        # Normalize shape - squeeze out singleton dimensions
        input_array = np.squeeze(input_array)

        # Ensure 2D or 3D (with color channels)
        if input_array.ndim not in [2, 3]:
            raise ValueError(f"Input must be 2D or 3D, got shape: {input_array.shape}")

        # Get dimensions
        if input_array.ndim == 2:
            img_h, img_w = input_array.shape
        else:
            img_h, img_w, _ = input_array.shape

        # Split in half
        half_width = img_w // 2
        
        if input_array.ndim == 2:
            left_half = input_array[:, :half_width]
            right_half = input_array[:, half_width:]
        else:
            left_half = input_array[:, :half_width, :]
            right_half = input_array[:, half_width:, :]

        # Flip right half horizontally
        right_half_flipped = np.flip(right_half, axis=1)

        if debug:
            self.print_log(f"Left half shape: {left_half.shape}", sample_id)
            self.print_log(f"Right half flipped shape: {right_half_flipped.shape}", sample_id)

        # Determine which side to return
        if side == "left":
            output_array = left_half
            side_label = "left"
        elif side == "right":
            output_array = right_half_flipped
            side_label = "right"
        elif side == "both":
            # Stack both halves vertically for visualization
            output_array = np.vstack([left_half, right_half_flipped])
            side_label = "both"
        else:
            raise ValueError(f"Invalid side parameter: {side}. Must be 'left', 'right', or 'both'")

        # Optional PNG save
        if save_png:
            # Use sample_id's output directory if available
            output_dir = getattr(sample_id, 'output_dir', '.')
            os.makedirs(output_dir, exist_ok=True)
            case_id = getattr(sample_id, 'case_id', 'unknown')
            output_path = os.path.join(output_dir, f"{case_id}_split_{side_label}.png")
            cv2.imwrite(output_path, output_array)

        # Prepare metadata
        new_metadata = None
        if input_image.metadata is not None:
            new_metadata = input_image.metadata.copy()
            
            # Update metadata for 2D images
            if output_array.ndim == 2:
                # Update direction matrix for 2D (2x2 = 4 elements)
                new_metadata['direction'] = [1.0, 0.0, 0.0, 1.0]
                # Update spacing for 2D
                if 'spacing' in new_metadata and len(new_metadata['spacing']) == 3:
                    new_metadata['spacing'] = new_metadata['spacing'][:2]
                # Update origin for 2D
                if 'origin' in new_metadata and len(new_metadata['origin']) == 3:
                    new_metadata['origin'] = new_metadata['origin'][:2]
            
            new_metadata.update({
                "side": side,
                "split_and_flipped": True
            })

        # Handle label array if present
        new_label_array = None
        if input_image.label_array is not None:
            label_array = input_image.label_array
            
            # Apply same processing to label array
            label_array = np.squeeze(label_array)
            
            if label_array.ndim == 2:
                left_label = label_array[:, :half_width]
                right_label = label_array[:, half_width:]
            else:
                left_label = label_array[:, :half_width, :]
                right_label = label_array[:, half_width:, :]
            
            right_label_flipped = np.flip(right_label, axis=1)
            
            if side == "left":
                new_label_array = left_label
            elif side == "right":
                new_label_array = right_label_flipped
            elif side == "both":
                new_label_array = np.vstack([left_label, right_label_flipped])

        self.print_log(f"Split and flip complete for {side} side(s)", sample_id)
        
        return SMImage(new_metadata, output_array, new_label_array)


if __name__ == "__main__":   
    tool = HalfAndFlip()
    asyncio.run(tool.main())
