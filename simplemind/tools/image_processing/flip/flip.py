"""
Tool Name: flip
=================================

Description:
    Flips an image over the x-axis, y-axis, or both depending on the parameter provided.

Parameters:            
    - input_image (SMImage): Input image to be flipped.
    - axis (str, optional): Which axis to flip - 'x', 'y', or 'both'. Default is 'x'.
        - 'x': Flip horizontally (left-right)
        - 'y': Flip vertically (up-down)  
        - 'both': Flip both horizontally and vertically
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - SMImage: The flipped image.
            
Example JSON Plan:
    "image_processing-flip": {
        "code": "flip.py",
        "input_image": "from input_image",
        "axis": "x",
        "debug": false
    }

Notes:
    - 'x' axis flip mirrors the image horizontally (left becomes right)
    - 'y' axis flip mirrors the image vertically (top becomes bottom)
    - 'both' applies both flips sequentially
    - Supports both 2D and 3D images with proper metadata handling.
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID


class Flip(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        axis: str = "x",
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

        # Apply flipping based on axis parameter
        if axis == "x":
            # Flip horizontally (left-right)
            output_array = np.flip(input_array, axis=1)
            flip_label = "horizontal"
        elif axis == "y":
            # Flip vertically (up-down)
            output_array = np.flip(input_array, axis=0)
            flip_label = "vertical"
        elif axis == "both":
            # Flip both horizontally and vertically
            output_array = np.flip(input_array, axis=1)  # First horizontal
            output_array = np.flip(output_array, axis=0)  # Then vertical
            flip_label = "both"
        else:
            raise ValueError(f"Invalid axis parameter: {axis}. Must be 'x', 'y', or 'both'")

        if debug:
            self.print_log(f"Applied {flip_label} flip", sample_id)
            self.print_log(f"Output shape: {output_array.shape}", sample_id)

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
                "flip_axis": axis,
                "flipped": True
            })

        # Handle label array if present
        new_label_array = None
        if input_image.label_array is not None:
            label_array = input_image.label_array
            
            # Apply same flipping to label array
            label_array = np.squeeze(label_array)
            
            if axis == "x":
                new_label_array = np.flip(label_array, axis=1)
            elif axis == "y":
                new_label_array = np.flip(label_array, axis=0)
            elif axis == "both":
                new_label_array = np.flip(label_array, axis=1)  # First horizontal
                new_label_array = np.flip(new_label_array, axis=0)  # Then vertical

        self.print_log(f"Image flip complete ({flip_label})", sample_id)
        
        return SMImage(new_metadata, output_array, new_label_array)


if __name__ == "__main__":
    tool = Flip()
    asyncio.run(tool.main())