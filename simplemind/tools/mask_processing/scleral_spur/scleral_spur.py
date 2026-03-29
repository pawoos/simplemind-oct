"""
Tool Name: scleral_spur
=================================

Description:
    Marks scleral spur coordinates on OCT images. Takes coordinates from image metadata or 
    explicit parameters and creates annotation masks for visualization. Follows SimpleMind's 
    pattern where original image stays unchanged and annotations go in label_array.

Parameters:            
    - input_image (SMImage): Input OCT image to mark scleral spur coordinates on. Default expects SMImage from left_half_image.
    - coordinate_source (SMImage, optional): SMImage containing scleral spur coordinates in metadata. Default is "from neural_net-ss".
    - scleral_spur_x (float, optional): X coordinate of scleral spur. Used when coordinate_source is not provided.
    - scleral_spur_y (float, optional): Y coordinate of scleral spur. Used when coordinate_source is not provided.
    - circle_radius (int, optional): Radius of the circle marker. Default is 1.
    - rectangle_size (int, optional): Size of the rectangle around the scleral spur. Default is 10.
    - draw_rectangle (bool, optional): Whether to draw the rectangle annotation. Default is False.
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - SMImage: The original image with scleral spur annotation mask in label_array.
            
Example JSON Plan (default - from neural_net-ss):
    "mask_processing-scleral_spur": {
        "code": "scleral_spur.py",
        "input_image": "from left_half_image",
        "coordinate_source": "from neural_net-ss",
        "circle_radius": 1,
        "rectangle_size": 10,
        "draw_rectangle": false,
        "debug": false
    }

Example with manual coordinates:
    "mask_processing-scleral_spur": {
        "code": "scleral_spur.py",
        "input_image": "from left_half_image",
        "scleral_spur_x": 150.5,
        "scleral_spur_y": 200.3,
        "circle_radius": 15,
        "rectangle_size": 75,
        "draw_rectangle": true
    }

Notes:
    - Default coordinate_source is "from neural_net-ss" which reads coordinates from neural_net-ss tool output metadata.
    - If coordinate_source is provided, reads 'scleral_spur_x' and 'scleral_spur_y' from that SMImage's metadata.
    - If coordinate_source is not provided, uses manual scleral_spur_x and scleral_spur_y parameters.
    - Default input_image expects SMImage from left_half_image tool.
    - Creates annotation mask with circle (value 1) and optional rectangle border (value 2) markers.
    - Rectangle annotation can be disabled by setting draw_rectangle to False.
    - Original grayscale image remains unchanged in pixel_array.
    - Annotation mask is stored in label_array for color overlay by view_image function.
    - Colors are handled automatically by view_image function based on mask values.
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID


class ScleralSpur(SMSampleProcessor):
    """Tool for marking scleral spur coordinates on OCT images."""

    def _create_scleral_spur_mask(self, image_shape: tuple, x: float, y: float, 
                                  circle_radius: int = 1, rectangle_size: int = 10, 
                                  draw_rectangle: bool = False) -> np.ndarray:
        """Create a mask array with scleral spur markers (circle and rectangle).
        
        Args:
            image_shape: Shape of the input image (C, Z, Y, X) format
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
        
        # For SMImage format (C, Z, Y, X), we work with the Y, X dimensions
        # Extract the 2D dimensions to work with
        if len(image_shape) == 4:
            # Standard SMImage format (C, Z, Y, X)
            c, z, h, w = image_shape
        elif len(image_shape) == 3:
            # Might be (Z, Y, X) or (C, Y, X)
            z, h, w = image_shape
        elif len(image_shape) == 2:
            # 2D (Y, X)
            h, w = image_shape
        else:
            raise ValueError(f"Unsupported image dimensions: {image_shape}")
        
        # Convert coordinates to integers and apply bounds checking
        x_int = int(round(x))
        y_int = int(round(y))
        
        # Bounds checking
        x_int = max(0, min(x_int, w - 1))  # x within [0, width-1]
        y_int = max(0, min(y_int, h - 1))  # y within [0, height-1]
        
        # Create a 2D mask array (same as working dimensions)
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        
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
        
        # Rectangle border (not filled) - create 2-pixel thick border only if enabled
        if draw_rectangle and rect_x2 > rect_x1 and rect_y2 > rect_y1:
            # Top and bottom borders
            mask_2d[rect_y1:min(rect_y1+2, h), rect_x1:rect_x2+1] = 2  # Top border
            mask_2d[max(rect_y2-1, 0):rect_y2+1, rect_x1:rect_x2+1] = 2  # Bottom border
            # Left and right borders
            mask_2d[rect_y1:rect_y2+1, rect_x1:min(rect_x1+2, w)] = 2  # Left border
            mask_2d[rect_y1:rect_y2+1, max(rect_x2-1, 0):rect_x2+1] = 2  # Right border
        
        # Convert back to the same format as input image using SMImage.normalize_dims
        # This ensures proper (C, Z, Y, X) format
        if len(image_shape) == 4:
            # Return as (C, Z, Y, X) format
            result_mask = np.zeros(image_shape, dtype=np.uint8)
            result_mask[0, 0, :, :] = mask_2d
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
        coordinate_source: SMImage = None,
        scleral_spur_x: float = None,
        scleral_spur_y: float = None,
        circle_radius: int = 1,
        rectangle_size: int = 10,
        draw_rectangle: bool = False,
        debug: bool = False,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None

        if debug:
            self.print_log(f"Input image shape: {input_image.pixel_array.shape}", sample_id)
            self.print_log(f"Coordinate source provided: {coordinate_source is not None}", sample_id)

        try:
            # Get coordinates based on the coordinate_source SMImage or manual parameters
            if coordinate_source is not None:
                # Use coordinates from the provided SMImage's metadata
                if coordinate_source.metadata is None:
                    raise ValueError("coordinate_source SMImage has no metadata")
                
                x_coord = coordinate_source.metadata.get('scleral_spur_x')
                y_coord = coordinate_source.metadata.get('scleral_spur_y')
                
                if x_coord is None or y_coord is None:
                    raise ValueError(f"No scleral spur coordinates found in coordinate_source metadata. "
                                   f"Expected 'scleral_spur_x' and 'scleral_spur_y' keys. "
                                   f"Available keys: {list(coordinate_source.metadata.keys())}")
                
                if debug:
                    self.print_log(f"Using coordinates from coordinate_source SMImage: ({x_coord}, {y_coord})", sample_id)
                    
            else:
                # Use manual coordinates
                if scleral_spur_x is None or scleral_spur_y is None:
                    raise ValueError("No coordinate_source provided and manual coordinates (scleral_spur_x, scleral_spur_y) are incomplete. "
                                   "Either provide coordinate_source SMImage or both manual coordinate parameters.")
                x_coord = scleral_spur_x
                y_coord = scleral_spur_y
                if debug:
                    self.print_log(f"Using manual coordinates: ({x_coord}, {y_coord})", sample_id)

            # Get input image array shape for mask creation
            input_array = input_image.pixel_array
            original_shape = input_array.shape
            
            if debug:
                self.print_log(f"Original image shape: {original_shape}", sample_id)
                self.print_log(f"Coordinates to mark: x={x_coord}, y={y_coord}", sample_id)
                self.print_log(f"Creating mask with circle radius {circle_radius} and rectangle size {rectangle_size}", sample_id)
                self.print_log(f"Draw rectangle: {draw_rectangle}", sample_id)

            # Create annotation mask using the helper method
            annotation_mask = self._create_scleral_spur_mask(
                original_shape, x_coord, y_coord, circle_radius, rectangle_size, draw_rectangle
            )
            
            if debug:
                self.print_log(f"Annotation mask shape: {annotation_mask.shape}, unique values: {np.unique(annotation_mask)}", sample_id)

            # Prepare metadata - keep original metadata structure
            new_metadata = None
            if input_image.metadata is not None:
                new_metadata = input_image.metadata.copy()
            else:
                new_metadata = {}
            
            # Add marking information to metadata
            new_metadata.update({
                "marked_scleral_spur": True,
                "marked_x": x_coord,
                "marked_y": y_coord,
                "coordinate_source_provided": coordinate_source is not None,
                "circle_radius": circle_radius,
                "rectangle_size": rectangle_size,
                "draw_rectangle": draw_rectangle
            })

            self.print_log(f"Scleral spur marking complete - coordinates: ({x_coord:.2f}, {y_coord:.2f})", sample_id)
            
            # Return SMImage with original pixel_array unchanged and annotation in label_array
            # This follows SimpleMind's pattern where view_image handles color overlay
            return SMImage(new_metadata, input_image.pixel_array, annotation_mask)

        except Exception as e:
            error_msg = f"Error in scleral spur marking: {str(e)}"
            self.print_error(error_msg, sample_id)
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    tool = ScleralSpur()
    asyncio.run(tool.main())