"""
Tool Name: image_mask
=================================

Description:
    Creates a mask image by filling a rectangular region defined by proportional
    bounds along the x, y, and z axes. Bounds are specified as proportions of 
    the respective dimension size (values in [0.0, 1.0]).

Parameters:            
    - input_image (SMImage): Input image (mask or image) used for dimensions.
    - x_upper_prop (float, optional): Upper x-axis proportion bound. Default = 1.
    - x_lower_prop (float, optional): Lower x-axis proportion bound. Default = 0.
    - y_upper_prop (float, optional): Upper y-axis proportion bound. Default = 1.
    - y_lower_prop (float, optional): Lower y-axis proportion bound. Default = 0.
    - z_upper_prop (float, optional): Upper z-axis proportion bound. Default = 1.
    - z_lower_prop (float, optional): Lower z-axis proportion bound. Default = 0.

Output:
    - SMImage: The generated rectangular mask image.

Example JSON Plan:
    "mask_generation-image_mask": {
        "code": "image_mask.py",
        "input_image": "from preprocessing-normalize",
        "x_upper_prop": 0.8,
        "x_lower_prop": 0.2,
        "y_upper_prop": 0.9,
        "y_lower_prop": 0.1,
        "z_upper_prop": 1.0,
        "": 0.0
    }
    
Notes:
    - z_upper_prop: proportion [0.0 - 1.0] of the image z-axis dimension that provides the upper z-axis bound of the output mask 
        - if not provided then the z_upper_prop is 1.0 by default (to cover the entire image with the mask)
    - z_lower_prop: proportion [0.0 - 1.0] of the image z-axis dimension that provides the lower z-axis bound of the output mask; 
        - if not provided then the z_lower_prop is 0 by default (to cover the entire image with the mask)
"""

import asyncio
import numpy as np
from typing import List

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

class ImageMask(SMSampleProcessor):

    @staticmethod
    def compute_operator(
        arr: np.ndarray,
        img_dim: int,
        dim: List[int],
        x_upper_prop: float,
        x_lower_prop: float,
        y_upper_prop: float,
        y_lower_prop: float,
        z_upper_prop: float,
        z_lower_prop: float,
    ) -> np.ndarray:

        new_arr = np.zeros_like(arr)

        min_y = int(y_lower_prop * (float(dim[1]) - 1))
        max_y = max(int(float(y_upper_prop) * (float(dim[1]) - 1)), min_y)

        min_x = int(float(x_lower_prop) * (float(dim[0]) - 1))
        max_x = max(int(float(x_upper_prop) * (float(dim[0]) - 1)), min_x)

        if str(img_dim) == "3":
            min_z = int(float(z_lower_prop) * (float(dim[2]) - 1))
            max_z = max(int(float(z_upper_prop) * (float(dim[2]) - 1)), min_z)
            new_arr[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = 1
        else:
            new_arr[min_y:max_y + 1, min_x:max_x + 1] = 1

        return new_arr

    async def execute(
        self,
        *,
        input_image: SMImage,
        x_upper_prop: float = 1,
        x_lower_prop: float = 0,
        y_upper_prop: float = 1,
        y_lower_prop: float = 0,
        z_upper_prop: float = 1,
        z_lower_prop: float = 0,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None

        arr = input_image.pixel_array
        metadata = input_image.metadata

        img_dim = arr.ndim
        if img_dim==3:
            dim = [
                arr.shape[2],
                arr.shape[1],
                arr.shape[0]
            ]
        else:
            dim = [
                arr.shape[1],
                arr.shape[0],
                0
            ]

        if (x_upper_prop < x_lower_prop):
            self.print_error(f"x_upper_prop ({x_upper_prop}) < x_lower_prop ({x_lower_prop}), returning empty mask", sample_id, warning=True)
            mask_arr = np.zeros_like(arr)
        elif (y_upper_prop < y_lower_prop):
            self.print_error(f"y_upper_prop ({y_upper_prop}) < y_lower_prop ({y_lower_prop}), returning empty mask", sample_id, warning=True)
            mask_arr = np.zeros_like(arr)
        elif (z_upper_prop < z_lower_prop):
            self.print_error(f"z_upper_prop ({z_upper_prop}) < z_lower_prop ({z_lower_prop}), returning empty mask", sample_id, warning=True)
            mask_arr = np.zeros_like(arr)
        else:            
            mask_arr = self.compute_operator(
                arr,
                img_dim,
                dim,
                x_upper_prop,
                x_lower_prop,
                y_upper_prop,
                y_lower_prop,
                z_upper_prop,
                z_lower_prop,
            )

        return SMImage(metadata, mask_arr, input_image.label_array)


if __name__ == "__main__":
    tool = ImageMask()
    asyncio.run(tool.main())
