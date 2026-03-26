"""
Tool Name: box_overlap
=================================

Description:
    Extracts a bounding box ROI from a single eye image based on cornea and iris mask overlap region.
    Finds the overlap between cornea and iris masks, then extracts a square region centered on that overlap.

Parameters:            
    - input_image (SMImage): Input grayscale image to extract ROI from.
    - cornea_mask (SMImage): Binary cornea mask.
    - iris_mask (SMImage): Binary iris mask.
    - box_size (int, optional): Size of the square ROI box in pixels. Default is 300.
    - max_dilation (int, optional): Maximum dilation iterations to find overlap. Default is 25.
    - save_png (bool, optional): Whether to save output as PNG file. Default is False.
    - debug (bool, optional): Enable debug logging. Default is False.
    - allow_failure (bool, optional): Allow returning None if no overlap found. Default is True.

Output:
    - SMImage: The extracted ROI image, or None if no overlap found and allow_failure=True.
            
Example JSON Plan:
    "image_processing-box_overlap": {
        "code": "box_overlap.py",
        "input_image": "from gray_image",
        "cornea_mask": "from cornea_mask",
        "iris_mask": "from iris_mask",
        "box_size": 300,
        "max_dilation": 25,
        "save_png": false,
        "debug": false,
        "allow_failure": true
    }

Notes:
    - Finds overlap between cornea and iris masks to determine ROI center.
    - If no initial overlap, dilates iris mask up to max_dilation iterations.
    - Extracts a square box_size x box_size region centered on the overlap.
    - Automatically adjusts box bounds to fit within image boundaries.
    - Returns None if no overlap found and allow_failure=True.
"""

import asyncio
import os
import numpy as np
import cv2

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID


def compute_box_from_masks(cornea_mask, iris_mask, pxls=300, max_dilation=100):
    """
    Given cornea and iris masks (binary 0/255),
    find a pxls x pxls box centered on the overlap region.
    Returns: top_left (x,y) in image coords, or None if failed.
    """
    # Ensure uint8
    cornea_mask = cornea_mask.astype(np.uint8)
    iris_mask = iris_mask.astype(np.uint8)

    # Initial overlap
    overlap_mask = cv2.bitwise_and(cornea_mask, iris_mask)
    overlap_points = np.column_stack(np.where(overlap_mask > 0))

    iteration = 0
    if len(overlap_points) == 0:
        # Dilate iris until overlap appears or max_dilation reached
        kernel = np.ones((5, 5), np.uint8)
        while len(overlap_points) == 0 and iteration <= max_dilation:
            dilated_iris = cv2.dilate(iris_mask, kernel, iterations=iteration)
            overlap_mask = cv2.bitwise_and(cornea_mask, dilated_iris)
            overlap_points = np.column_stack(np.where(overlap_mask > 0))
            iteration += 1

    if len(overlap_points) == 0:
        # Failed to find overlap
        return None

    # Find centermost overlap point
    centroid = np.mean(overlap_points, axis=0).astype(int)
    distances = np.linalg.norm(overlap_points - centroid, axis=1)
    centermost_point = overlap_points[np.argmin(distances)]  # (row, col) = (y, x)

    # Center pxls x pxls box on this point
    cy, cx = int(centermost_point[0]), int(centermost_point[1])
    top_left_x = cx - pxls // 2
    top_left_y = cy - pxls // 2

    return (top_left_x, top_left_y)


def fix_box_bounds(top_left, box_size, img_w, img_h):
    """
    Ensure the box defined by top_left and box_size fits within the image.
    Returns (fixed_top_left, fixed_bottom_right) in image coords.
    """
    w, h = box_size, box_size
    x, y = top_left

    # Fix negative x/y
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    # Initial bottom-right
    br_x = x + w
    br_y = y + h

    # If box extends past right/bottom edge, shift back
    if br_x > img_w:
        shift = br_x - img_w
        x = max(0, x - shift)
        br_x = x + w

    if br_y > img_h:
        shift = br_y - img_h
        y = max(0, y - shift)
        br_y = y + h

    # Final clamp (for cases where image is smaller than box)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    br_x = max(0, min(br_x, img_w))
    br_y = max(0, min(br_y, img_h))

    return (x, y), (br_x, br_y)


class BoxOverlap(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        cornea_mask: SMImage,
        iris_mask: SMImage,
        box_size: int = 300,
        max_dilation: int = 25,
        save_png: bool = False,
        debug: bool = False,
        allow_failure: bool = True,
        sample_id: SMSampleID
    ) -> SMImage:
        if input_image is None or cornea_mask is None or iris_mask is None:
            return None

        # Extract arrays
        gray_img = input_image.pixel_array
        cornea_array = cornea_mask.pixel_array
        iris_array = iris_mask.pixel_array

        if debug:
            self.print_log(f"Input shapes - gray: {gray_img.shape}, cornea: {cornea_array.shape}, iris: {iris_array.shape}", sample_id)

        # Normalize shapes - squeeze out singleton dimensions
        gray_img = np.squeeze(gray_img)
        cornea_array = np.squeeze(cornea_array)
        iris_array = np.squeeze(iris_array)

        # Ensure grayscale (2D)
        if gray_img.ndim == 3:
            if gray_img.shape[2] == 3:
                gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
            elif gray_img.shape[2] == 1:
                gray_img = gray_img[:, :, 0]
            else:
                raise ValueError(f"Unexpected gray image shape: {gray_img.shape}")
        
        if gray_img.ndim != 2:
            raise ValueError(f"Gray image must be 2D after processing, got shape: {gray_img.shape}")

        # Ensure masks are 2D
        if cornea_array.ndim != 2:
            raise ValueError(f"Cornea mask must be 2D, got shape: {cornea_array.shape}")
        if iris_array.ndim != 2:
            raise ValueError(f"Iris mask must be 2D, got shape: {iris_array.shape}")

        # Check shape consistency
        if gray_img.shape != cornea_array.shape or gray_img.shape != iris_array.shape:
            raise ValueError(
                f"Shape mismatch - gray {gray_img.shape}, "
                f"cornea {cornea_array.shape}, iris {iris_array.shape}"
            )

        if debug:
            self.print_log(f"Normalized shapes - gray: {gray_img.shape}, cornea: {cornea_array.shape}, iris: {iris_array.shape}", sample_id)

        # Find bounding box based on overlap
        img_h, img_w = gray_img.shape
        top_left = compute_box_from_masks(cornea_array, iris_array, pxls=box_size, max_dilation=max_dilation)

        if top_left is None:
            if allow_failure:
                side_label = input_image.metadata.get('side', 'unknown') if input_image.metadata else 'unknown'
                case_id = getattr(sample_id, 'case_id', 'unknown')
                self.print_log(f"WARNING: {case_id} {side_label} - Failed to find overlap between cornea and iris masks - returning None", sample_id)
                return None
            else:
                raise ValueError("Failed to find overlap between cornea and iris masks")

        # Fix bounds to ensure box fits within image
        (tl_x, tl_y), (br_x, br_y) = fix_box_bounds(top_left, box_size, img_w, img_h)

        if debug:
            self.print_log(f"Bounding box - top-left: ({tl_x}, {tl_y}), bottom-right: ({br_x}, {br_y})", sample_id)

        # Crop the ROI
        cropped = gray_img[tl_y:br_y, tl_x:br_x]

        if debug:
            self.print_log(f"Cropped ROI shape: {cropped.shape}", sample_id)

        # Prepare metadata
        new_metadata = None
        if input_image.metadata is not None:
            new_metadata = input_image.metadata.copy()
            
            # Ensure metadata is 2D
            if cropped.ndim == 2:
                # Update direction matrix for 2D (2x2 = 4 elements)
                new_metadata['direction'] = [1.0, 0.0, 0.0, 1.0]
                # Update spacing for 2D
                if 'spacing' in new_metadata and len(new_metadata['spacing']) == 3:
                    new_metadata['spacing'] = new_metadata['spacing'][:2]
                # Update origin for 2D
                if 'origin' in new_metadata and len(new_metadata['origin']) == 3:
                    new_metadata['origin'] = new_metadata['origin'][:2]
            
            # Get side from metadata (set by split_and_flip agent)
            side_label = new_metadata.get('side', 'unknown')
            
            new_metadata.update({
                "box_size": box_size,
                "top_left": (tl_x, tl_y),
                "bottom_right": (br_x, br_y),
                "roi_extracted": True
            })
            
            if debug:
                case_id = getattr(sample_id, 'case_id', 'unknown')
                self.print_log(f"Metadata - case_id: {case_id}, side: {side_label}", sample_id)

        # Optional PNG save
        if save_png:
            # Use sample_id's output directory if available
            output_dir = getattr(sample_id, 'output_dir', '.')
            os.makedirs(output_dir, exist_ok=True)
            case_id = getattr(sample_id, 'case_id', 'unknown')
            side_label = new_metadata.get('side', 'unknown') if new_metadata else 'unknown'
            output_path = os.path.join(output_dir, f"{case_id}_roi_{side_label}.png")
            cv2.imwrite(output_path, cropped)
            if debug:
                self.print_log(f"Saved PNG to: {output_path}", sample_id)

        self.print_log(f"ROI extraction complete - box size {box_size}x{box_size}", sample_id)
        
        return SMImage(new_metadata, cropped, None)


if __name__ == "__main__":
    tool = BoxOverlap()
    asyncio.run(tool.main())
