"""
Tool Name: crop
=================================

Description:
    Crops an input image based on specified parameters. Supports two cropping modes:
    1. Offset-based: Remove pixels from top, bottom, left, right edges
    2. Region-based: Specify start coordinates and crop dimensions

Parameters:            
    - input_image (SMImage): Input image to be cropped.
    - crop_top (int, optional): Number of pixels to remove from top edge.
    - crop_bottom (int, optional): Number of pixels to remove from bottom edge.
    - crop_left (int, optional): Number of pixels to remove from left edge.
    - crop_right (int, optional): Number of pixels to remove from right edge.
    - start_x (int, optional): Starting x coordinate for region-based cropping.
    - start_y (int, optional): Starting y coordinate for region-based cropping.
    - start_z (int, optional): Starting z coordinate for 3D region-based cropping.
    - crop_width (int, optional): Width of the crop region.
    - crop_height (int, optional): Height of the crop region.
    - crop_depth (int, optional): Depth of the crop region (for 3D images).
    - square_size (int, optional): Size for perfect square crop (sets both width and height).
    - center_x (int, optional): X coordinate for center of square crop (used with square_size).
    - center_y (int, optional): Y coordinate for center of square crop (used with square_size).
    - side (str, optional): Side to crop from - "left" (default) or "right". When "right", flips cropping parameters.
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - SMImage: The cropped image with metadata containing crop region coordinates.
      The metadata includes:
      - cropped: True (indicates this image has been cropped)
      - original_shape: Shape of the input image before cropping
      - crop_region: Dictionary containing:
        - start_x, start_y: Top-left coordinates of the crop region
        - end_x, end_y: Bottom-right coordinates of the crop region  
        - width, height: Dimensions of the cropped region
        - start_z, end_z, depth: 3D coordinates (for 3D images only)
            
Example JSON Plan (Offset-based):
    "image_processing-crop": {
        "code": "crop.py",
        "input_image": "from input_image",
        "crop_top": 50,
        "crop_bottom": 50,
        "crop_left": 100,
        "crop_right": 100,
        "side": "left",
        "debug": false
    }

Example JSON Plan (Right-side cropping):
    "image_processing-crop": {
        "code": "crop.py",
        "input_image": "from input_image",
        "crop_top": 50,
        "crop_bottom": 50,
        "crop_left": 100,
        "crop_right": 100,
        "side": "right",
        "debug": false
    }

Example JSON Plan (Region-based):
    "image_processing-crop": {
        "code": "crop.py",
        "input_image": "from input_image",
        "start_x": 100,
        "start_y": 50,
        "crop_width": 300,
        "crop_height": 200,
        "debug": false
    }

Example JSON Plan (Square crop):
    "image_processing-crop": {
        "code": "crop.py",
        "input_image": "from input_image",
        "square_size": 100,
        "center_x": 250,
        "center_y": 200,
        "debug": false
    }

Notes:
    - If square_size is provided, it creates a perfect square crop and takes priority over crop_width/crop_height.
    - If region parameters (start_x, start_y, crop_width, crop_height) are provided, they take priority over offset parameters.
    - For offset-based cropping, any unspecified edge will not be cropped (default 0).
    - The 'side' parameter controls cropping direction: "left" (default) crops normally, "right" flips left/right parameters and start_x coordinates.
    - When side="right": crop_left becomes crop_right, crop_right becomes crop_left, and start_x is calculated from the right edge.
    - Automatically handles bounds checking to ensure crop region is within image boundaries.
    - Supports both 2D and 3D images with proper metadata handling.
    - The output metadata includes crop_region coordinates that can be used by subsequent tools to map the cropped region back to the original image coordinates.
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID


class Crop(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        crop_top: int = 0,
        crop_bottom: int = 0,
        crop_left: int = 0,
        crop_right: int = 0,
        start_x: int = None,
        start_y: int = None,
        start_z: int = None,
        crop_width: int = None,
        crop_height: int = None,
        crop_depth: int = None,
        square_size: int = None,
        center_x: int = None,
        center_y: int = None,
        side: str = "left",
        debug: bool = False,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None

        # Extract input
        input_array = input_image.pixel_array

        if debug:
            self.print_log(f"Input shape: {input_array.shape}", sample_id)
            self.print_log(f"Cropping from side: {side}", sample_id)

        # Normalize shape - squeeze out singleton dimensions
        input_array = np.squeeze(input_array)

        # Ensure 2D or 3D (with color channels)
        if input_array.ndim not in [2, 3]:
            raise ValueError(f"Input must be 2D or 3D, got shape: {input_array.shape}")

        # Get image dimensions for side parameter processing
        if input_array.ndim == 2:
            img_h, img_w = input_array.shape
        else:
            if len(input_array.shape) == 3 and input_array.shape[2] <= 4:  # Color channels
                img_h, img_w, _ = input_array.shape
            else:  # 3D volume
                _, img_h, img_w = input_array.shape

        # Handle side parameter - flip parameters if cropping from right
        if side.lower() == "right":
            if debug:
                self.print_log("Flipping parameters for right-side cropping", sample_id)
            
            # Flip offset-based parameters
            crop_left, crop_right = crop_right, crop_left
            
            # Flip region-based parameters
            if start_x is not None:
                # Convert start_x from left-based to right-based coordinate
                if crop_width is not None:
                    start_x = img_w - start_x - crop_width
                else:
                    start_x = img_w - start_x
            
            # Flip center_x for square crops
            if center_x is not None:
                center_x = img_w - center_x
                
            if debug:
                self.print_log(f"After flipping - crop_left: {crop_left}, crop_right: {crop_right}", sample_id)
                if start_x is not None:
                    self.print_log(f"After flipping - start_x: {start_x}", sample_id)
                if center_x is not None:
                    self.print_log(f"After flipping - center_x: {center_x}", sample_id)

        # Determine cropping mode and calculate crop bounds
        if square_size is not None:
            # Square crop mode - takes highest priority
            if debug:
                self.print_log(f"Using square crop mode - size: {square_size}", sample_id)
            
            # Calculate center point
            if center_x is not None and center_y is not None:
                # Use provided center
                cx, cy = center_x, center_y
            else:
                # Use image center
                cx, cy = img_w // 2, img_h // 2

            # Calculate square bounds
            half_size = square_size // 2
            start_x = cx - half_size
            start_y = cy - half_size
            end_x = start_x + square_size
            end_y = start_y + square_size

            # Bounds checking and adjustment
            if start_x < 0:
                shift = -start_x
                start_x = 0
                end_x = min(img_w, square_size)
            elif end_x > img_w:
                shift = end_x - img_w
                end_x = img_w
                start_x = max(0, img_w - square_size)
            
            if start_y < 0:
                shift = -start_y
                start_y = 0
                end_y = min(img_h, square_size)
            elif end_y > img_h:
                shift = end_y - img_h
                end_y = img_h
                start_y = max(0, img_h - square_size)

            if debug:
                self.print_log(f"Square crop - center: ({cx}, {cy}), bounds: ({start_x}, {start_y}) to ({end_x}, {end_y})", sample_id)
                actual_width = end_x - start_x
                actual_height = end_y - start_y
                if actual_width != square_size or actual_height != square_size:
                    self.print_log(f"Warning: Requested square size {square_size}x{square_size}, actual crop size {actual_width}x{actual_height} due to image boundaries", sample_id)

            # Apply square cropping
            if input_array.ndim == 2:
                cropped_array = input_array[start_y:end_y, start_x:end_x]
            else:
                if len(input_array.shape) == 3 and input_array.shape[2] <= 4:  # Color channels
                    cropped_array = input_array[start_y:end_y, start_x:end_x, :]
                else:  # 3D volume
                    cropped_array = input_array[:, start_y:end_y, start_x:end_x]

        elif start_x is not None or start_y is not None or crop_width is not None or crop_height is not None:
            # Region-based cropping mode
            if debug:
                self.print_log("Using region-based cropping mode", sample_id)
            
            # Get 3D dimension if needed
            if input_array.ndim == 2:
                img_d = 1
            else:
                if len(input_array.shape) == 3 and input_array.shape[2] <= 4:  # Color channels
                    img_d = input_array.shape[2]
                else:  # 3D volume
                    img_d = input_array.shape[0]

            # Set defaults for region parameters
            start_x = start_x if start_x is not None else 0
            start_y = start_y if start_y is not None else 0
            start_z = start_z if start_z is not None else 0
            crop_width = crop_width if crop_width is not None else img_w - start_x
            crop_height = crop_height if crop_height is not None else img_h - start_y
            crop_depth = crop_depth if crop_depth is not None else img_d - start_z

            # Calculate end coordinates
            end_x = start_x + crop_width
            end_y = start_y + crop_height
            end_z = start_z + crop_depth

            # Bounds checking
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            start_z = max(0, start_z)
            end_x = min(img_w, end_x)
            end_y = min(img_h, end_y)
            end_z = min(img_d, end_z)

            if debug:
                self.print_log(f"Region crop - start: ({start_x}, {start_y}, {start_z}), end: ({end_x}, {end_y}, {end_z})", sample_id)

            # Apply region-based cropping
            if input_array.ndim == 2:
                cropped_array = input_array[start_y:end_y, start_x:end_x]
            else:
                if len(input_array.shape) == 3 and input_array.shape[2] <= 4:  # Color channels
                    cropped_array = input_array[start_y:end_y, start_x:end_x, :]
                else:  # 3D volume
                    cropped_array = input_array[start_z:end_z, start_y:end_y, start_x:end_x]

        else:
            # Offset-based cropping mode
            if debug:
                self.print_log("Using offset-based cropping mode", sample_id)
            
            # Calculate crop bounds from offsets
            start_x = crop_left
            end_x = img_w - crop_right
            start_y = crop_top
            end_y = img_h - crop_bottom

            # Bounds checking
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(img_w, max(start_x + 1, end_x))  # Ensure at least 1 pixel width
            end_y = min(img_h, max(start_y + 1, end_y))  # Ensure at least 1 pixel height

            if debug:
                self.print_log(f"Offset crop - removing top: {crop_top}, bottom: {crop_bottom}, left: {crop_left}, right: {crop_right}", sample_id)
                self.print_log(f"Crop bounds - start: ({start_x}, {start_y}), end: ({end_x}, {end_y})", sample_id)

            # Apply offset-based cropping
            if input_array.ndim == 2:
                cropped_array = input_array[start_y:end_y, start_x:end_x]
            else:
                if len(input_array.shape) == 3 and input_array.shape[2] <= 4:  # Color channels
                    cropped_array = input_array[start_y:end_y, start_x:end_x, :]
                else:  # 3D volume
                    cropped_array = input_array[:, start_y:end_y, start_x:end_x]

        if debug:
            self.print_log(f"Cropped shape: {cropped_array.shape}", sample_id)

        # Prepare metadata
        new_metadata = None
        if input_image.metadata is not None:
            new_metadata = input_image.metadata.copy()
            
            # Update metadata for 2D images
            if cropped_array.ndim == 2:
                # Update direction matrix for 2D (2x2 = 4 elements)
                new_metadata['direction'] = [1.0, 0.0, 0.0, 1.0]
                # Update spacing for 2D
                if 'spacing' in new_metadata and len(new_metadata['spacing']) == 3:
                    new_metadata['spacing'] = new_metadata['spacing'][:2]
                # Update origin for 2D
                if 'origin' in new_metadata and len(new_metadata['origin']) == 3:
                    new_metadata['origin'] = new_metadata['origin'][:2]
            
            # Add crop region coordinates to metadata
            crop_region = {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "width": end_x - start_x,
                "height": end_y - start_y
            }
            
            # Add 3D coordinates if applicable
            if input_array.ndim == 3 and not (len(input_array.shape) == 3 and input_array.shape[2] <= 4):
                crop_region.update({
                    "start_z": start_z if 'start_z' in locals() else 0,
                    "end_z": end_z if 'end_z' in locals() else input_array.shape[0],
                    "depth": (end_z if 'end_z' in locals() else input_array.shape[0]) - (start_z if 'start_z' in locals() else 0)
                })
            
            new_metadata.update({
                "cropped": True,
                "original_shape": input_array.shape,
                "crop_region": crop_region
            })
            
            if debug:
                self.print_log(f"Added crop region metadata: {crop_region}", sample_id)

        # Handle label array if present
        new_label_array = None
        if input_image.label_array is not None:
            label_array = input_image.label_array
            
            # Apply same cropping to label array
            label_array = np.squeeze(label_array)
            
            # Apply the same cropping logic to label array
            if square_size is not None:  # Square-based
                if label_array.ndim == 2:
                    new_label_array = label_array[start_y:end_y, start_x:end_x]
                else:
                    if len(label_array.shape) == 3 and label_array.shape[2] <= 4:  # Color channels
                        new_label_array = label_array[start_y:end_y, start_x:end_x, :]
                    else:  # 3D volume
                        new_label_array = label_array[:, start_y:end_y, start_x:end_x]
            elif start_x is not None and crop_width is not None:  # Region-based
                if label_array.ndim == 2:
                    new_label_array = label_array[start_y:end_y, start_x:end_x]
                else:
                    if len(label_array.shape) == 3 and label_array.shape[2] <= 4:  # Color channels
                        new_label_array = label_array[start_y:end_y, start_x:end_x, :]
                    else:  # 3D volume
                        new_label_array = label_array[start_z:end_z, start_y:end_y, start_x:end_x]
            else:  # Offset-based
                if label_array.ndim == 2:
                    new_label_array = label_array[start_y:end_y, start_x:end_x]
                else:
                    if len(label_array.shape) == 3 and label_array.shape[2] <= 4:  # Color channels
                        new_label_array = label_array[start_y:end_y, start_x:end_x, :]
                    else:  # 3D volume
                        new_label_array = label_array[:, start_y:end_y, start_x:end_x]

        # Log crop region metadata
        if new_metadata and 'crop_region' in new_metadata:
            crop_region = new_metadata['crop_region']
            self.print_log(f"Crop region coordinates - start: ({crop_region['start_x']}, {crop_region['start_y']}), end: ({crop_region['end_x']}, {crop_region['end_y']})", sample_id)
            self.print_log(f"Crop region dimensions - width: {crop_region['width']}, height: {crop_region['height']}", sample_id)
            
            # Log 3D coordinates if present
            if 'start_z' in crop_region:
                self.print_log(f"Crop region 3D - start_z: {crop_region['start_z']}, end_z: {crop_region['end_z']}, depth: {crop_region['depth']}", sample_id)

        self.print_log(f"Image crop complete - new shape: {cropped_array.shape}", sample_id)
        
        return SMImage(new_metadata, cropped_array, new_label_array)


if __name__ == "__main__":
    tool = Crop()
    asyncio.run(tool.main())
