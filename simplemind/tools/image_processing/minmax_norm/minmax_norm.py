"""
Tool Name: minmax_norm
=================================

Description:
    Apply min-max normalization to an SMImage. The normalized image is min value of 0.0 and max value of 1.0.

Parameters:            
    - input_image (SMImage): Input image to be normalized.

Output:
    - SMImage: The normalized image.
            
Example JSON Plan:
    "minmax_norm": {
        "code": "minmax_norm.py",
        "input_image": "from read_sm_image",
    }

Notes:
    - If all values are uniform in the input image, then the result is all 0.
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage

class MinMaxNorm(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage       
    ) -> SMImage:

        if input_image is None:
            return None

        min_val = input_image.pixel_array.min()
        max_val = input_image.pixel_array.max()
        if min_val == max_val:
            new_pixel_array = np.zeros_like(input_image.pixel_array)
        else:
            new_pixel_array = (input_image.pixel_array - min_val)/(max_val - min_val)
        
        return SMImage(input_image.metadata, new_pixel_array, input_image.label_array)

if __name__ == "__main__":   
    tool = MinMaxNorm()
    asyncio.run(tool.main())
