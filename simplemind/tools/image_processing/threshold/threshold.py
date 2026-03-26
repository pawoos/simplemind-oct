"""
Tool Name: threshold
=================================

Description:
    Performs thresholding on an SMImage and returns a binary mask as an SMImage.

Parameters:            
    - input_image (SMImage): Input image to be thresholded.
    - lower_threshold (float, optional): Lower threshold bound.
    - upper_threshold (float, optional): Upper threshold bound.

Output:
    - SMImage: The thresholded mask.
            
Example JSON Plan:
    "threshold": {
        "code": "threshold_tool.py",
        "context": "./threshold_tool/",
        "input_image": "from clahe",
        "upper_threshold": 0.4
    }

Notes:
    - Thresholds use >= and <=.
    - lower_threshold and upper_threshold are both optional, but at least one must be provided.
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage

class Threshold(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        lower_threshold: float = None,
        upper_threshold: float = None,
    ) -> SMImage:   

        if input_image is None:
            return None

        if lower_threshold is not None and upper_threshold is not None and (lower_threshold > upper_threshold):
            self.print_error(f"Lower bound is > than upper bound: {lower_threshold} > {upper_threshold}", warning=True)
            return None

        if upper_threshold is None and lower_threshold is None:
            self.print_error("No thresholds provided", warning=True)
            return None

        thresholded_arr = None
        if lower_threshold is not None:
            thresholded_arr = input_image.pixel_array >= lower_threshold

        if upper_threshold is not None:
            if thresholded_arr is not None:
                thresholded_arr = np.logical_and(thresholded_arr, input_image.pixel_array <= upper_threshold)
            else:
                thresholded_arr = input_image.pixel_array <= upper_threshold

        thresholded_arr = thresholded_arr.astype(int)
        
        return  SMImage(input_image.metadata, thresholded_arr, input_image.label_array)


if __name__ == "__main__":   
    tool = Threshold()
    asyncio.run(tool.main())
