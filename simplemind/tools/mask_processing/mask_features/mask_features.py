"""
Tool Name: mask_features
=================================

Description:
    Computes a dictonary of mask features.

Parameters:            
    - input_mask (SMImage): Input binary mask image.

Output:
    - dictionary of mask features: {
        centroid:
    }

Example JSON Plan:
    "bounding_box": {
        "code": "bounding_box.py",
        "input_image": "from mask_processing-morph_close",
        "z_upper_offset": 5,
        "offset_unit": "mm"
    }
    
Notes:
    - 
"""

import asyncio
import numpy as np
from typing import List
from scipy.ndimage import center_of_mass

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

@staticmethod
def binary_centroid(arr: np.ndarray) -> tuple[int, ...]:
    """
    Compute the centroid of nonzero elements in an array,
    treating all nonzero values as 1, and return integer coordinates.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array (any shape, any dtype).
    
    Returns
    -------
    tuple[int, ...]
        Integer coordinates of the centroid (nearest voxel/pixel).
    
    Raises
    ------
    ValueError
        If the array contains no nonzero elements.
    """
    # Get coordinates of nonzero elements
    positions = np.argwhere(arr != 0)
    if positions.size == 0:
        raise ValueError("Array has no nonzero elements")

    # Compute mean along each axis
    mean_coords = positions.mean(axis=0)

    # Round to nearest integer
    int_coords = tuple(int(round(c)) for c in mean_coords)
    return int_coords

class MaskFeatures(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_mask: SMImage,
        sample_id: SMSampleID
    ) -> dict:

        if input_mask is None:
            return None

        result = {}
        result['centroid'] = binary_centroid(input_mask.pixel_array)
        
        # self.print_log(f"mask_features={result}", sample_id)

        return result


if __name__ == "__main__":
    tool = MaskFeatures()
    asyncio.run(tool.main())
