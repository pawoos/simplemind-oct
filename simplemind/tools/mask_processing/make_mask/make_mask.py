"""
Tool Name: make_mask
=================================

Description:
    Generate an SMImage binary mask with a list of points.

Parameters:            
    - target_image (SMImage): The mask takes the shape of its pixel_array.
        The mask also takes its metadata.
    - mask_points (list, optional): List of points where each point is a tuple (z,y,x) or list. Default = None (no points added).

Output:
    - SMImage: The binary mask as pixel_array.
            
Example JSON Plan:
    "mask_points": {
        "code": "make_mask.py",
        "chunk": "make_mask",
        "target_image": "from medsam2",
        "mask_points": [[1, 150, 150], [1, 300, 300]],
        "final_output": true
    }

Notes:
    - Binary mask has only 0s and 1s.
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

class MakeMask(SMSampleProcessor):

    async def execute(
        self,
        *,
        target_image: SMImage,
        mask_points: list = None,
        sample_id: SMSampleID
    ) -> SMImage:

        if target_image is None:
            return None

        if target_image is not None:
            mask_shape = target_image.pixel_array.shape

        new_mask = self.make_binary_mask(mask_shape, mask_points, sample_id)
        mask_metadata = target_image.metadata
        
        return SMImage(mask_metadata, new_mask)

    def make_binary_mask(self, shape, points, sample_id):
        """Core mask creation."""
        mask = np.zeros(shape, dtype=np.uint8)
        for p in points:
            tp = tuple(p)
            try:
                mask[tp] = 1
            except IndexError:
                self.print_error(f"Point {tp} is out of bounds for shape {shape}", sample_id, warning=True)
        return mask


if __name__ == "__main__":   
    tool = MakeMask()
    asyncio.run(tool.main())
