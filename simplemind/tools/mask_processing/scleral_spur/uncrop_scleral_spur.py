"""
Tool Name: uncrop_scleral_spur
=================================

Description:
    Takes scleral spur coordinates from a cropped image detection and translates them back to 
    full-sized image space, then annotates the scleral spur location on the original image.
    Uses crop_region metadata from crop tool and scleral_spur coordinates from oct_ss_detection tool.

Parameters:            
    - input_image (SMImage): Original full-sized image to annotate.
    - cropped_detection (SMImage): Image with scleral spur detection results containing both crop_region and scleral_spur coordinates.
    - circle_radius (int, optional): Radius of the circle marker. Default is 3.
    - rectangle_size (int, optional): Size of the rectangle around the scleral spur. Default is 15.
    - draw_rectangle (bool, optional): Whether to draw the rectangle annotation. Default is False.
    - side (str, optional): Side the crop was taken from - "left" (default) or "right". Default is "left".
    - flip (bool, optional): Whether to flip the scleral spur coordinates over half-image y-axis. Default is False.
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - SMImage: The original image with scleral spur location annotated in label_array and coordinates in metadata.
            
Example JSON Plan:
    "mask_processing-uncrop_scleral_spur": {
        "code": "uncrop_scleral_spur.py",
        "input_image": "from input_image",
        "cropped_detection": "from neural_net-oct_ss_detection",
        "circle_radius": 3,
        "rectangle_size": 15,
        "draw_rectangle": false,
        "side": "left",
        "flip": false,
        "debug": false
    }

Example with right-side flip transformation:
    "mask_processing-uncrop_scleral_spur": {
        "code": "uncrop_scleral_spur.py",
        "input_image": "from input_image",
        "cropped_detection": "from neural_net-oct_ss_detection",
        "circle_radius": 3,
        "rectangle_size": 15,
        "draw_rectangle": true,
        "side": "right",
        "flip": true,
        "debug": true
    }

Notes:
    - Reads crop_region metadata to determine coordinate translation offset.
    - Reads scleral_spur_x and scleral_spur_y coordinates from detection results.
    - Translates coordinates from cropped space to full image space.
    - Creates annotation mask with circle (value 1) and optional rectangle border (value 2) markers.
    - Original image remains unchanged in pixel_array.
    - Annotation marker is stored in label_array for color overlay.
    - Translated coordinates are added to output metadata as 'full_image_scleral_spur_x' and 'full_image_scleral_spur_y'.
    - Coordinate transformations are applied independently:
      * flip=True: Flips scleral spur coordinates over y-axis of the half-image (using half_width as flip axis)
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


class UncropScleralSpur(SMSampleProcessor):
    """Tool for translating scleral spur coordinates from cropped space to full image space and annotating."""

    def _create_scleral_spur_mask(self, image_shape: tuple, x: float, y: float, 
                                  circle_radius: int = 3, rectangle_size: int = 15, 
                                  draw_rectangle: bool = True, debug: bool = False, sample_id = None) -> np.ndarray:
        """Create a mask array with scleral spur markers (circle and rectangle).
        
        Args:
            image_shape: Shape of the input image
            x: X coordinate (width dimension)
            y: Y coordinate (height dimension)
            circle_radius: Radius of the circle marker
            rectangle_size: Half-size of the rectangle around the scleral spur
            draw_rectangle: Whether to draw the rectangle annotation
            
        Returns:
            np.ndarray: Mask array with same shape as input, containing:
                        - 0: background
                        - 1: circle marker (filled)
                        - 2: rectangle border
        """
        
        # Handle different image dimensions - similar to uncrop.py pattern
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
        
        # Convert coordinates to integers and apply bounds checking
        x_int = int(round(x))
        y_int = int(round(y))
        
        # Bounds checking
        x_int = max(0, min(x_int, w - 1))  # x within [0, width-1]
        y_int = max(0, min(y_int, h - 1))  # y within [0, height-1]
        
        if debug:
            self.print_log(f"Marker center after bounds checking: ({x_int}, {y_int})", sample_id)
        
        # Create a 2D mask array (same as scleral_spur pattern)
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        
        if debug:
            self.print_log(f"Created 2D mask with shape: {mask_2d.shape}", sample_id)
        
        # Create circle mask
        y_coords, x_coords = np.ogrid[:h, :w]
        circle_mask = (x_coords - x_int)**2 + (y_coords - y_int)**2 <= circle_radius**2
        
        # Create rectangle mask with bounds checking
        rect_x1 = max(0, x_int - rectangle_size)
        rect_y1 = max(0, y_int - rectangle_size)
        rect_x2 = min(w - 1, x_int + rectangle_size)
        rect_y2 = min(h - 1, y_int + rectangle_size)
        
        # Set mask values: 1 for circle (filled), 2 for rectangle border (if enabled)
        mask_2d[circle_mask] = 1  # Circle gets value 1
        
        if debug:
            circle_pixels = np.count_nonzero(circle_mask)
            self.print_log(f"Circle: center=({x_int}, {y_int}), radius={circle_radius}, pixels set: {circle_pixels}", sample_id)
        
        # Rectangle border (not filled) - create 2-pixel thick border only if enabled
        if draw_rectangle and rect_x2 > rect_x1 and rect_y2 > rect_y1:
            # Top and bottom borders
            mask_2d[rect_y1:min(rect_y1+2, h), rect_x1:rect_x2+1] = 2  # Top border
            mask_2d[max(rect_y2-1, 0):rect_y2+1, rect_x1:rect_x2+1] = 2  # Bottom border
            # Left and right borders
            mask_2d[rect_y1:rect_y2+1, rect_x1:min(rect_x1+2, w)] = 2  # Left border
            mask_2d[rect_y1:rect_y2+1, max(rect_x2-1, 0):rect_x2+1] = 2  # Right border
            
            if debug:
                rect_pixels = np.count_nonzero(mask_2d == 2)
                self.print_log(f"Rectangle: bounds=[{rect_x1},{rect_y1}] to [{rect_x2},{rect_y2}], border pixels set: {rect_pixels}", sample_id)
        
        if debug:
            self.print_log(f"2D mask after markers - unique values: {np.unique(mask_2d)}, non-zero: {np.count_nonzero(mask_2d)}", sample_id)
            # Check if marker was actually created
            if np.count_nonzero(mask_2d) == 0:
                self.print_log(f"WARNING: No pixels were set in marker! Check coordinates and bounds.", sample_id)
        
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
        cropped_detection: SMImage,
        circle_radius: int = 3,
        rectangle_size: int = 15,
        draw_rectangle: bool = False,
        side: str = "left",
        flip: bool = False,
        debug: bool = False,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None
            
        if cropped_detection is None:
            raise ValueError("cropped_detection is required to extract crop_region and scleral_spur coordinates")

        if debug:
            self.print_log(f"Input image shape: {input_image.pixel_array.shape}", sample_id)
            self.print_log(f"Cropped detection provided: {cropped_detection is not None}", sample_id)

        try:
            # Extract metadata from cropped detection image
            if cropped_detection.metadata is None:
                raise ValueError("cropped_detection has no metadata")
            
            # Get crop region metadata
            crop_region = cropped_detection.metadata.get('crop_region')
            if crop_region is None:
                raise ValueError("No crop_region found in cropped_detection metadata. "
                               "This tool requires an image that was processed by the crop tool.")
            
            # Get scleral spur coordinates
            spur_x_cropped = cropped_detection.metadata.get('scleral_spur_x')
            spur_y_cropped = cropped_detection.metadata.get('scleral_spur_y')
            
            if spur_x_cropped is None or spur_y_cropped is None:
                raise ValueError("No scleral_spur_x or scleral_spur_y found in cropped_detection metadata. "
                               "This tool requires an image that was processed by the oct_ss_detection tool.")
            
            if debug:
                self.print_log(f"Found crop_region metadata: {crop_region}", sample_id)
                self.print_log(f"Found scleral spur coordinates in cropped space: ({spur_x_cropped}, {spur_y_cropped})", sample_id)
                self.print_log(f"Side: {side}, Flip: {flip}", sample_id)
                
            # Handle side and flip parameters for crop region metadata
            if flip:
                if debug:
                    self.print_log("Flipping crop_region metadata coordinates", sample_id)
                
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
                
                # Create a copy of crop_region to modify
                crop_region = crop_region.copy()
                
                # Flip the crop region coordinates over half-image y-axis
                half_image_width = img_w // 2
                original_start_x = crop_region['start_x']
                original_end_x = crop_region['end_x']
                
                # Flip the crop region coordinates
                crop_region['start_x'] = half_image_width - original_end_x
                crop_region['end_x'] = half_image_width - original_start_x
                crop_region['width'] = crop_region['end_x'] - crop_region['start_x']
                
                if debug:
                    self.print_log(f"Original crop_region: start_x={original_start_x}, end_x={original_end_x}", sample_id)
                    self.print_log(f"Flipped crop_region: start_x={crop_region['start_x']}, end_x={crop_region['end_x']}", sample_id)

            # Validate required crop region fields
            required_fields = ['start_x', 'start_y', 'end_x', 'end_y']
            missing_fields = [field for field in required_fields if field not in crop_region]
            if missing_fields:
                raise ValueError(f"crop_region metadata is missing required fields: {missing_fields}")

            # Step 1: Translate scleral spur coordinates from cropped space to half-image space
            # (using the potentially flipped crop_region coordinates)
            half_image_spur_x = spur_x_cropped + crop_region['start_x']
            half_image_spur_y = spur_y_cropped + crop_region['start_y']
            
            if debug:
                self.print_log(f"Step 1 - Translated from cropped to half-image space: ({half_image_spur_x}, {half_image_spur_y})", sample_id)

            # Step 2: Apply side transformation if right side (map from half-image to full image)
            if side.lower() == "right":
                if debug:
                    self.print_log("Step 2 - Applying right-side coordinate shift (half-image -> full image space)", sample_id)
                
                # Add half the image width to map from right-half space to full image space
                if 'img_w' not in locals():
                    # Get image width if not already calculated
                    input_array = input_image.pixel_array
                    original_shape = input_array.shape
                    
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
                
                half_width = img_w // 2
                full_spur_x = half_image_spur_x + half_width
                
                if debug:
                    self.print_log(f"After adding half width ({half_width}): spur_x={full_spur_x}", sample_id)
            else:
                # Left side - no additional offset needed
                full_spur_x = half_image_spur_x
            
            full_spur_y = half_image_spur_y
            transformation_applied = flip or side.lower() == "right"
            
            if debug:
                self.print_log(f"Final scleral spur coordinates: ({full_spur_x}, {full_spur_y})", sample_id)
            if debug:
                self.print_log(f"Crop region offset: ({crop_region['start_x']}, {crop_region['start_y']})", sample_id)
                self.print_log(f"Translated scleral spur coordinates to full image space: ({full_spur_x}, {full_spur_y})", sample_id)

            # Get input image array shape for mask creation
            input_array = input_image.pixel_array
            original_shape = input_array.shape
            
            if debug:
                self.print_log(f"Original image shape: {original_shape}", sample_id)
                self.print_log(f"Circle radius: {circle_radius}, Rectangle size: {rectangle_size}, Draw rectangle: {draw_rectangle}", sample_id)

            # Create scleral spur marker annotation mask
            spur_marker_mask = self._create_scleral_spur_mask(
                original_shape, full_spur_x, full_spur_y, circle_radius, rectangle_size, draw_rectangle, debug, sample_id
            )
            
            if debug:
                self.print_log(f"Created spur marker mask - shape: {spur_marker_mask.shape}, dtype: {spur_marker_mask.dtype}", sample_id)
                self.print_log(f"Spur marker mask unique values: {np.unique(spur_marker_mask)}", sample_id)
                self.print_log(f"Non-zero spur marker pixels: {np.count_nonzero(spur_marker_mask)}", sample_id)
            
            # Start with existing label_array if available, otherwise create new
            if input_image.label_array is not None:
                annotation_mask = input_image.label_array.copy()
                if debug:
                    self.print_log(f"Starting with existing label_array - shape: {annotation_mask.shape}, unique values: {np.unique(annotation_mask)}", sample_id)
            else:
                annotation_mask = np.zeros(original_shape, dtype=np.int32)
                if debug:
                    self.print_log(f"Created new annotation mask - shape: {annotation_mask.shape}", sample_id)
            
            # Ensure mask is integer type for label_array compatibility
            if annotation_mask.dtype != np.int32:
                annotation_mask = annotation_mask.astype(np.int32)
                if debug:
                    self.print_log(f"Converted mask to int32 for label_array compatibility", sample_id)
            
            # Add scleral spur markers to existing annotations
            # Circle gets value 1, rectangle border gets value 2 (same as scleral_spur.py)
            circle_locations = spur_marker_mask == 1
            rectangle_locations = spur_marker_mask == 2
            
            annotation_mask[circle_locations] = 1  # Circle marker
            if draw_rectangle:
                annotation_mask[rectangle_locations] = 2  # Rectangle border
            
            if debug:
                self.print_log(f"Added scleral spur markers - circle (value 1) and rectangle (value 2)", sample_id)
                self.print_log(f"Final annotation mask unique values: {np.unique(annotation_mask)}", sample_id)
                self.print_log(f"Final annotation mask non-zero count: {np.count_nonzero(annotation_mask)}", sample_id)
                
                # Check specific marker location
                if annotation_mask.ndim == 4:
                    marker_check = annotation_mask[0, :, :, 0] if annotation_mask.shape[3] == 1 else annotation_mask[0, 0, :, :]
                else:
                    marker_check = annotation_mask.squeeze() if annotation_mask.ndim > 2 else annotation_mask
                
                # Find where circle and rectangle markers exist
                circle_locations = np.where(marker_check == 1)
                rectangle_locations = np.where(marker_check == 2)
                
                if len(circle_locations[0]) > 0:
                    self.print_log(f"Circle marker (value 1) found at {len(circle_locations[0])} locations", sample_id)
                    self.print_log(f"First few circle locations (y,x): {list(zip(circle_locations[0][:5], circle_locations[1][:5]))}", sample_id)
                else:
                    self.print_log(f"WARNING: No circle marker (value 1) found in final mask!", sample_id)
                    
                if draw_rectangle and len(rectangle_locations[0]) > 0:
                    self.print_log(f"Rectangle marker (value 2) found at {len(rectangle_locations[0])} locations", sample_id)
                elif draw_rectangle:
                    self.print_log(f"WARNING: No rectangle marker (value 2) found in final mask!", sample_id)

            # Prepare metadata - keep original metadata and add scleral spur information
            new_metadata = None
            if input_image.metadata is not None:
                new_metadata = input_image.metadata.copy()
            else:
                new_metadata = {}
            
            # Add translated scleral spur coordinates and annotation information to metadata
            new_metadata.update({
                "full_image_scleral_spur_x": full_spur_x,
                "full_image_scleral_spur_y": full_spur_y,
                "scleral_spur_annotated": True,
                "cropped_scleral_spur_x": spur_x_cropped,
                "cropped_scleral_spur_y": spur_y_cropped,
                "crop_region_used": crop_region.copy(),
                "circle_radius": circle_radius,
                "rectangle_size": rectangle_size,
                "draw_rectangle": draw_rectangle,
                "side": side,
                "flip": flip,
                "coordinate_transformed": transformation_applied,
                "scleral_spur_detected": cropped_detection.metadata.get('scleral_spur_detected', True),
                "box_size": cropped_detection.metadata.get('box_size')
            })

            # Log the results
            self.print_log(f"Scleral spur coordinates translated from cropped ({spur_x_cropped:.2f}, {spur_y_cropped:.2f}) "
                         f"to full image ({full_spur_x:.2f}, {full_spur_y:.2f})", sample_id)
            self.print_log(f"Scleral spur markers annotated - circle radius: {circle_radius}, rectangle size: {rectangle_size}, draw rectangle: {draw_rectangle}", sample_id)
            
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
            error_msg = f"Error in scleral spur uncrop annotation: {str(e)}"
            self.print_error(error_msg, sample_id)
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    tool = UncropScleralSpur()
    asyncio.run(tool.main())