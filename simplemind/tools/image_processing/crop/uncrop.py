"""
Tool Name: uncrop
=================================

Description:
    Takes crop_region metadata from a cropped image and annotates the boundary of the 
    cropped region on the original full-sized image. Creates a label mask showing the 
    rectangular or square crop boundary that was applied by the crop tool.

Parameters:            
    - input_image (SMImage): Original full-sized image to annotate.
    - cropped_image (SMImage): Cropped image containing crop_region metadata from crop tool.
    - border_thickness (int, optional): Thickness of the crop boundary annotation. Default is 2.
    - fill_region (bool, optional): Whether to fill the entire crop region or just draw the border. Default is False (border only).
    - side (str, optional): Side the crop was taken from - "left" (default) or "right". Default is "left".
    - flip (bool, optional): Whether to flip the crop region coordinates over y-axis. Default is False.
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - SMImage: The original image with crop region boundary annotation in label_array.
            
Example JSON Plan:
    "image_processing-uncrop": {
        "code": "uncrop.py",
        "input_image": "from input_image",
        "cropped_image": "from image_processing-crop",
        "border_thickness": 2,
        "fill_region": false,
        "side": "left",
        "flip": false,
        "debug": false
    }

Example with left-side flip only:
    "image_processing-uncrop": {
        "code": "uncrop.py",
        "input_image": "from input_image", 
        "cropped_image": "from image_processing-crop",
        "border_thickness": 2,
        "fill_region": false,
        "side": "left",
        "flip": true,
        "debug": true
    }

Example with right-side shift only:
    "image_processing-uncrop": {
        "code": "uncrop.py",
        "input_image": "from input_image", 
        "cropped_image": "from image_processing-crop",
        "border_thickness": 2,
        "fill_region": false,
        "side": "right",
        "flip": false,
        "debug": true
    }

Example with both flip and right-side shift:
    "image_processing-uncrop": {
        "code": "uncrop.py",
        "input_image": "from input_image", 
        "cropped_image": "from image_processing-crop",
        "border_thickness": 3,
        "fill_region": true,
        "side": "right",
        "flip": true,
        "debug": true
    }

Notes:
    - Reads crop_region metadata from cropped_image to determine boundary coordinates.
    - Creates annotation mask with crop boundary (value 1) and optional filled region (value 2).
    - Original image remains unchanged in pixel_array.
    - Annotation mask is stored in label_array for color overlay by view_image function.
    - Supports both 2D and 3D images with proper coordinate handling.
    - If fill_region is True, the entire crop region is filled with value 2, and border gets value 1.
    - If fill_region is False, only the border is drawn with value 1.
    - Coordinate transformations are applied independently:
      * flip=True: Flips crop_region coordinates over y-axis of the half-image (using half_width as flip axis)
      * side="right": Adds half the image width to x-coordinates
      * Both can be combined for complex coordinate mappings
    - Transformation combinations:
      * side="left", flip=False: No transformation (default)
      * side="left", flip=True: Only half-image y-axis flip (new_x = half_width - old_x)
      * side="right", flip=False: Only half-width x-shift (new_x = old_x + half_width)
      * side="right", flip=True: Both half-image flip and half-width x-shift
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID


class Uncrop(SMSampleProcessor):
    """Tool for annotating crop region boundaries on original images."""

    def _create_crop_boundary_mask(self, image_shape: tuple, crop_region: dict, 
                                   border_thickness: int = 2, fill_region: bool = False, debug: bool = False, sample_id = None) -> np.ndarray:
        """Create a mask array with crop region boundary annotation.
        
        Args:
            image_shape: Shape of the original image
            crop_region: Dictionary containing crop coordinates (start_x, start_y, end_x, end_y, etc.)
            border_thickness: Thickness of the boundary lines
            fill_region: Whether to fill the entire crop region or just draw the border
            
        Returns:
            np.ndarray: Mask array with same shape as input, containing:
                        - 0: background
                        - 1: crop boundary
                        - 2: filled crop region (if fill_region=True)
        """
        
        # Extract crop coordinates
        start_x = crop_region['start_x']
        start_y = crop_region['start_y'] 
        end_x = crop_region['end_x']
        end_y = crop_region['end_y']
        
        # Handle different image dimensions - fix for proper dimension interpretation
        if len(image_shape) == 4:
            # For 4D arrays like (1, 1505, 1920, 1), need to identify Y, X dimensions
            c, dim1, dim2, dim3 = image_shape
            
            # If last dimension is 1, likely (C, Y, X, Z) format
            if dim3 == 1:
                h, w = dim1, dim2  # (C, Y, X, Z) format
            else:
                # Standard SMImage format (C, Z, Y, X)
                h, w = dim2, dim3
                
        elif len(image_shape) == 3:
            # Might be (Z, Y, X) or (C, Y, X)
            z, h, w = image_shape
        elif len(image_shape) == 2:
            # 2D (Y, X)
            h, w = image_shape
        else:
            raise ValueError(f"Unsupported image dimensions: {image_shape}")
            
        if debug:
            self.print_log(f"Interpreted dimensions - height: {h}, width: {w} from shape {image_shape}", sample_id)
        
        # Bounds checking
        start_x = max(0, min(start_x, w - 1))
        start_y = max(0, min(start_y, h - 1))
        end_x = max(start_x + 1, min(end_x, w))
        end_y = max(start_y + 1, min(end_y, h))
        
        if debug:
            self.print_log(f"After bounds checking - start: ({start_x}, {start_y}), end: ({end_x}, {end_y})", sample_id)
        
        # Create a 2D mask array (same as scleral_spur pattern)
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        
        if debug:
            self.print_log(f"Created 2D mask with shape: {mask_2d.shape}", sample_id)
        
        # Fill the entire crop region if requested
        if fill_region:
            mask_2d[start_y:end_y, start_x:end_x] = 2
            if debug:
                self.print_log(f"Filled region [{start_y}:{end_y}, {start_x}:{end_x}] with value 2", sample_id)
        
        # Draw the border with specified thickness
        half_thickness = border_thickness // 2
        
        if debug:
            self.print_log(f"Drawing border with thickness {border_thickness}, half_thickness {half_thickness}", sample_id)
        
        # Top border
        top_start = max(0, start_y - half_thickness)
        top_end = min(h, start_y + border_thickness - half_thickness)
        mask_2d[top_start:top_end, start_x:end_x] = 1
        if debug:
            self.print_log(f"Top border: [{top_start}:{top_end}, {start_x}:{end_x}]", sample_id)
        
        # Bottom border  
        bottom_start = max(0, end_y - border_thickness + half_thickness)
        bottom_end = min(h, end_y + half_thickness)
        mask_2d[bottom_start:bottom_end, start_x:end_x] = 1
        if debug:
            self.print_log(f"Bottom border: [{bottom_start}:{bottom_end}, {start_x}:{end_x}]", sample_id)
        
        # Left border
        left_start = max(0, start_x - half_thickness)
        left_end = min(w, start_x + border_thickness - half_thickness)
        mask_2d[start_y:end_y, left_start:left_end] = 1
        if debug:
            self.print_log(f"Left border: [{start_y}:{end_y}, {left_start}:{left_end}]", sample_id)
        
        # Right border
        right_start = max(0, end_x - border_thickness + half_thickness)
        right_end = min(w, end_x + half_thickness)
        mask_2d[start_y:end_y, right_start:right_end] = 1
        if debug:
            self.print_log(f"Right border: [{start_y}:{end_y}, {right_start}:{right_end}]", sample_id)
            self.print_log(f"2D mask after borders - unique values: {np.unique(mask_2d)}, non-zero: {np.count_nonzero(mask_2d)}", sample_id)
        
        # Convert back to the same format as input image - handle 4D case properly
        if len(image_shape) == 4:
            # Return as original format - need to handle (C, Y, X, Z) vs (C, Z, Y, X)
            result_mask = np.zeros(image_shape, dtype=np.uint8)
            c, dim1, dim2, dim3 = image_shape
            
            if dim3 == 1:  # (C, Y, X, Z) format
                result_mask[0, :, :, 0] = mask_2d
                if debug:
                    self.print_log(f"Assigned 2D mask to result_mask[0, :, :, 0] for (C, Y, X, Z) format", sample_id)
            else:  # (C, Z, Y, X) format
                result_mask[0, 0, :, :] = mask_2d
                if debug:
                    self.print_log(f"Assigned 2D mask to result_mask[0, 0, :, :] for (C, Z, Y, X) format", sample_id)
            return result_mask
        elif len(image_shape) == 3:
            # Return as (Z, Y, X) format - assume first dim is Z
            result_mask = np.zeros(image_shape, dtype=np.uint8)
            result_mask[0, :, :] = mask_2d
            return result_mask
        else:
            # Return as 2D - let SMImage.normalize_dims handle it
            return mask_2d

    async def execute(
        self,
        *,
        input_image: SMImage,
        cropped_image: SMImage,
        border_thickness: int = 2,
        fill_region: bool = False,
        side: str = "left",
        flip: bool = False,
        debug: bool = False,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None
            
        if cropped_image is None:
            raise ValueError("cropped_image is required to extract crop_region metadata")

        if debug:
            self.print_log(f"Input image shape: {input_image.pixel_array.shape}", sample_id)
            self.print_log(f"Cropped image provided: {cropped_image is not None}", sample_id)

        try:
            # Extract crop_region metadata from cropped image
            if cropped_image.metadata is None:
                raise ValueError("cropped_image has no metadata")
            
            crop_region = cropped_image.metadata.get('crop_region')
            if crop_region is None:
                raise ValueError("No crop_region found in cropped_image metadata. "
                               "This tool requires an image that was processed by the crop tool.")
            
            if debug:
                self.print_log(f"Found crop_region metadata: {crop_region}", sample_id)
                self.print_log(f"Side: {side}, Flip: {flip}", sample_id)
                
            # Handle side and flip parameters
            transformation_applied = False
            if side.lower() == "right" or flip:
                if debug:
                    self.print_log("Applying coordinate transformation to crop_region", sample_id)
                
                # Get input image width for coordinate transformation
                input_array = input_image.pixel_array
                original_shape = input_array.shape
                
                # Determine image width from shape
                if len(original_shape) == 4:
                    c, dim1, dim2, dim3 = original_shape
                    if dim3 == 1:  # (C, Y, X, Z) format
                        img_w = dim2
                    else:  # (C, Z, Y, X) format
                        img_w = dim3
                elif len(original_shape) == 3:
                    z, h, img_w = original_shape
                elif len(original_shape) == 2:
                    h, img_w = original_shape
                else:
                    raise ValueError(f"Unsupported image dimensions: {original_shape}")
                
                if debug:
                    self.print_log(f"Image width determined as: {img_w}", sample_id)
                    self.print_log(f"Original crop_region: start_x={crop_region['start_x']}, end_x={crop_region['end_x']}", sample_id)
                
                # Create a copy of crop_region to modify
                crop_region = crop_region.copy()
                original_start_x = crop_region['start_x']
                original_end_x = crop_region['end_x']
                
                # Step 1: Apply flip transformation if requested
                if flip:
                    if debug:
                        self.print_log("Applying y-axis flip transformation over half-image width", sample_id)
                    
                    # Flip coordinates over y-axis of the half image
                    # For half-image flip: new_x = half_width - old_x
                    half_width = img_w // 2
                    flipped_start_x = half_width - original_end_x
                    flipped_end_x = half_width - original_start_x
                    
                    crop_region['start_x'] = flipped_start_x
                    crop_region['end_x'] = flipped_end_x
                    
                    if debug:
                        self.print_log(f"Half-image width: {half_width}", sample_id)
                        self.print_log(f"After half-image y-axis flip: start_x={flipped_start_x}, end_x={flipped_end_x}", sample_id)
                
                # Step 2: Apply side transformation if right side
                if side.lower() == "right":
                    if debug:
                        self.print_log("Applying right-side coordinate shift", sample_id)
                    
                    # Add half the image width to map to original coordinate system
                    half_width = img_w // 2
                    crop_region['start_x'] += half_width
                    crop_region['end_x'] += half_width
                    
                    if debug:
                        self.print_log(f"After adding half width ({half_width}): start_x={crop_region['start_x']}, end_x={crop_region['end_x']}", sample_id)
                
                # Update width to maintain consistency
                crop_region['width'] = crop_region['end_x'] - crop_region['start_x']
                transformation_applied = True
                
                if debug:
                    self.print_log(f"Final transformed crop_region: {crop_region}", sample_id)
                
            # Validate required crop region fields
            required_fields = ['start_x', 'start_y', 'end_x', 'end_y']
            missing_fields = [field for field in required_fields if field not in crop_region]
            if missing_fields:
                raise ValueError(f"crop_region metadata is missing required fields: {missing_fields}")

            # Get input image array shape for mask creation (if not already extracted above)
            if 'input_array' not in locals():
                input_array = input_image.pixel_array
            original_shape = input_array.shape
            
            if debug:
                self.print_log(f"Original image shape: {original_shape}", sample_id)
                self.print_log(f"Crop region coordinates: start=({crop_region['start_x']}, {crop_region['start_y']}), "
                             f"end=({crop_region['end_x']}, {crop_region['end_y']})", sample_id)
                self.print_log(f"Border thickness: {border_thickness}, Fill region: {fill_region}", sample_id)

            # Create crop boundary annotation mask
            annotation_mask = self._create_crop_boundary_mask(
                original_shape, crop_region, border_thickness, fill_region, debug, sample_id
            )
            
            if debug:
                self.print_log(f"Created annotation mask - shape: {annotation_mask.shape}, dtype: {annotation_mask.dtype}", sample_id)
                self.print_log(f"Annotation mask unique values: {np.unique(annotation_mask)}", sample_id)
                self.print_log(f"Non-zero mask pixels: {np.count_nonzero(annotation_mask)}", sample_id)
                
                # Log some sample mask values at the border locations
                if annotation_mask.ndim >= 2:
                    mask_2d = annotation_mask.squeeze() if annotation_mask.ndim > 2 else annotation_mask
                    self.print_log(f"Sample mask values at crop region:", sample_id)
                    self.print_log(f"  Top-left corner ({crop_region['start_x']}, {crop_region['start_y']}): {mask_2d[crop_region['start_y'], crop_region['start_x']] if crop_region['start_y'] < mask_2d.shape[0] and crop_region['start_x'] < mask_2d.shape[1] else 'out of bounds'}", sample_id)
                    self.print_log(f"  Top border center: {mask_2d[crop_region['start_y'], (crop_region['start_x'] + crop_region['end_x'])//2] if crop_region['start_y'] < mask_2d.shape[0] and (crop_region['start_x'] + crop_region['end_x'])//2 < mask_2d.shape[1] else 'out of bounds'}", sample_id)
            
            # Ensure mask is integer type for label_array compatibility
            if annotation_mask.dtype != np.int32:
                annotation_mask = annotation_mask.astype(np.int32)
                if debug:
                    self.print_log(f"Converted mask to int32 for label_array compatibility", sample_id)

            # Prepare metadata - keep original metadata and add uncrop information
            new_metadata = None
            if input_image.metadata is not None:
                new_metadata = input_image.metadata.copy()
            else:
                new_metadata = {}
            
            # Add uncrop annotation information to metadata
            new_metadata.update({
                "uncrop_annotated": True,
                "annotated_crop_region": crop_region.copy(),
                "border_thickness": border_thickness,
                "fill_region": fill_region,
                "side": side,
                "flip": flip,
                "coordinate_transformed": transformation_applied,
                "original_cropped_shape": cropped_image.metadata.get('original_shape') if cropped_image.metadata else None
            })

            # Log crop region coordinates
            self.print_log(f"Crop region boundary annotated - start: ({crop_region['start_x']}, {crop_region['start_y']}), "
                         f"end: ({crop_region['end_x']}, {crop_region['end_y']})", sample_id)
            self.print_log(f"Crop region dimensions - width: {crop_region['width']}, height: {crop_region['height']}", sample_id)
            
            # Log 3D coordinates if present
            if 'start_z' in crop_region:
                self.print_log(f"Crop region 3D - start_z: {crop_region['start_z']}, end_z: {crop_region['end_z']}, depth: {crop_region['depth']}", sample_id)

            self.print_log(f"Uncrop annotation complete - border thickness: {border_thickness}, fill region: {fill_region}", sample_id)
            
            # Return SMImage with original pixel_array unchanged and annotation in label_array
            result_image = SMImage(new_metadata, input_image.pixel_array, annotation_mask)
            
            if debug:
                self.print_log(f"Final SMImage created:", sample_id)
                self.print_log(f"  pixel_array shape: {result_image.pixel_array.shape}", sample_id)
                self.print_log(f"  label_array shape: {result_image.label_array.shape if result_image.label_array is not None else 'None'}", sample_id)
                if result_image.label_array is not None:
                    self.print_log(f"  label_array dtype: {result_image.label_array.dtype}", sample_id)
                    self.print_log(f"  label_array unique values: {np.unique(result_image.label_array)}", sample_id)
                    self.print_log(f"  label_array non-zero count: {np.count_nonzero(result_image.label_array)}", sample_id)
            
            return result_image

        except Exception as e:
            error_msg = f"Error in uncrop annotation: {str(e)}"
            self.print_error(error_msg, sample_id)
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    tool = Uncrop()
    asyncio.run(tool.main())