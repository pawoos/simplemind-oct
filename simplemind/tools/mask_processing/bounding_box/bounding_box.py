"""
Tool Name: bounding_box
=================================

Description:
    Computes a bounding box mask around nonzero regions of an SMImage mask.
    Supports voxel, mm, and length-fraction offset units, as well as slice-wise bounding boxes.

Parameters:            
    - input_image (SMImage): Input binary mask image.
    - z_upper_offset (float, optional): Offset added to the upper z coordinate.
    - z_lower_offset (float, optional): Offset added to the lower z coordinate.
    - y_upper_offset (float, optional): Offset added to the upper y coordinate.
    - y_lower_offset (float, optional): Offset added to the lower y coordinate.
    - x_upper_offset (float, optional): Offset added to the upper x coordinate.
    - x_lower_offset (float, optional): Offset added to the lower x coordinate.
    - offset_unit (str, optional): One of "mm", "voxels" (default), or "length_fraction".
    - slice_wise_bounding_box (bool, optional): If True, compute slice-wise 2D bounding boxes.
    - axis (str, optional): Axis for slice-wise bounding boxes ("z", "y", or "x"). Default = "z".

Output:
    - SMImage: The bounding box mask image.

Example JSON Plan:
    "bounding_box": {
        "code": "bounding_box.py",
        "input_image": "from mask_processing-morph_close",
        "z_upper_offset": 5,
        "offset_unit": "mm"
    }
    
Notes:
    - If the input mask is empty then the output mask will also be empty.
"""

import asyncio
import numpy as np
from typing import List

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage


class BoundingBox(SMSampleProcessor):

    @staticmethod
    def get_bounding_box(
        arr: np.ndarray,
        x_upper_offset: float = 0, x_lower_offset: float = 0,
        z_upper_offset: float = 0, z_lower_offset: float = 0,
        y_upper_offset: float = 0, y_lower_offset: float = 0,
        spacing: List[float] = [1, 1, 1],
    ) -> np.ndarray:
        z, y, x = np.where(arr == 1)
        bounding_box = np.zeros(arr.shape, dtype=arr.dtype)

        if len(z) != 0:
            min_z = max(np.min(z) + int(z_lower_offset / spacing[2]), 0)
            max_z = min(np.max(z) + int(z_upper_offset / spacing[2]), arr.shape[0] - 1)
            min_y = max(np.min(y) + int(y_lower_offset / spacing[1]), 0)
            max_y = min(np.max(y) + int(y_upper_offset / spacing[1]), arr.shape[1] - 1)
            min_x = max(np.min(x) + int(x_lower_offset / spacing[0]), 0)
            max_x = min(np.max(x) + int(x_upper_offset / spacing[0]), arr.shape[2] - 1)

            if min_z <= max_z and min_y <= max_y and min_x <= max_x:
                bounding_box[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = 1

        return bounding_box

    @staticmethod
    def get_bounding_box_fraction(
        arr: np.ndarray,
        x_upper_offset: float = 0, x_lower_offset: float = 0,
        z_upper_offset: float = 0, z_lower_offset: float = 0,
        y_upper_offset: float = 0, y_lower_offset: float = 0,
    ) -> np.ndarray:
        z, y, x = np.where(arr == 1)
        bounding_box = np.zeros(arr.shape, dtype=arr.dtype)

        if len(z) != 0:
            z_len = np.max(z) - np.min(z)
            min_z = max(np.min(z) + int(z_lower_offset * z_len), 0)
            max_z = min(np.max(z) + int(z_upper_offset * z_len), arr.shape[0] - 1)

            y_len = np.max(y) - np.min(y)
            min_y = max(np.min(y) + int(y_lower_offset * y_len), 0)
            max_y = min(np.max(y) + int(y_upper_offset * y_len), arr.shape[1] - 1)

            x_len = np.max(x) - np.min(x)
            min_x = max(np.min(x) + int(x_lower_offset * x_len), 0)
            max_x = min(np.max(x) + int(x_upper_offset * x_len), arr.shape[2] - 1)

            if min_z <= max_z and min_y <= max_y and min_x <= max_x:
                bounding_box[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = 1

        return bounding_box

    @staticmethod
    def get_bounding_box_2d(
        arr: np.ndarray,
        x_upper_offset: float = 0, x_lower_offset: float = 0,
        y_upper_offset: float = 0, y_lower_offset: float = 0,
        spacing: List[float] = [1, 1, 1],
        axis: str = 'z',
    ) -> np.ndarray:
        bounding_box = np.zeros(arr.shape, dtype=arr.dtype)
        Z, Y, X = np.where(arr == 1)

        match axis:
            case 'z':
                for z in range(np.min(Z), np.max(Z) + 1):
                    y, x = np.where(arr[z] == 1)
                    if len(y) == 0: 
                        continue
                    min_y = max(np.min(y) + int(y_lower_offset / spacing[1]), 0)
                    max_y = min(np.max(y) + int(y_upper_offset / spacing[1]), arr.shape[1] - 1)
                    min_x = max(np.min(x) + int(x_lower_offset / spacing[0]), 0)
                    max_x = min(np.max(x) + int(x_upper_offset / spacing[0]), arr.shape[2] - 1)
                    if min_y <= max_y and min_x <= max_x:
                        bounding_box[z, min_y:max_y + 1, min_x:max_x + 1] = 1

            case 'y':
                for y in range(np.min(Y), np.max(Y) + 1):
                    z, x = np.where(arr[:, y, :] == 1)
                    if len(z) == 0:
                        continue
                    min_z = max(np.min(z) + int(y_lower_offset / spacing[2]), 0)
                    max_z = min(np.max(z) + int(y_upper_offset / spacing[2]), arr.shape[0] - 1)
                    min_x = max(np.min(x) + int(x_lower_offset / spacing[0]), 0)
                    max_x = min(np.max(x) + int(x_upper_offset / spacing[0]), arr.shape[2] - 1)
                    if min_z <= max_z and min_x <= max_x:
                        bounding_box[min_z:max_z + 1, y, min_x:max_x + 1] = 1

            case 'x':
                for x in range(np.min(X), np.max(X) + 1):
                    z, y = np.where(arr[:, :, x] == 1)
                    if len(z) == 0:
                        continue
                    min_z = max(np.min(z) + int(y_lower_offset / spacing[2]), 0)
                    max_z = min(np.max(z) + int(y_upper_offset / spacing[2]), arr.shape[0] - 1)
                    min_y = max(np.min(y) + int(x_lower_offset / spacing[1]), 0)
                    max_y = min(np.max(y) + int(x_upper_offset / spacing[1]), arr.shape[1] - 1)
                    if min_z <= max_z and min_y <= max_y:
                        bounding_box[min_z:max_z + 1, min_y:max_y + 1, x] = 1

        return bounding_box

    async def execute(
        self,
        *,
        input_image: SMImage,
        z_upper_offset: float = 0,
        z_lower_offset: float = 0,
        y_upper_offset: float = 0,
        y_lower_offset: float = 0,
        x_upper_offset: float = 0,
        x_lower_offset: float = 0,
        offset_unit: str = "voxels",
        slice_wise_bounding_box: bool = False,
        axis: str = "z",
    ) -> SMImage:

        if input_image is None:
            return None

        arr = input_image.pixel_array
        if arr.ndim == 2:  # ensure 3D
            arr = np.expand_dims(arr, axis=0)

        if offset_unit == "mm":
            spacing = input_image.metadata.get("spacing", [1, 1, 1])
        else:
            spacing = [1, 1, 1]

        if np.all(arr == 0):
            BB_array = np.zeros(arr.shape)
        elif offset_unit == "length_fraction":
            BB_array = self.get_bounding_box_fraction(
                arr, x_upper_offset, x_lower_offset,
                z_upper_offset, z_lower_offset,
                y_upper_offset, y_lower_offset,
            )
        elif slice_wise_bounding_box:
            BB_array = self.get_bounding_box_2d(
                arr, x_upper_offset, x_lower_offset,
                y_upper_offset, y_lower_offset,
                spacing=spacing, axis=axis,
            )
        else:
            BB_array = self.get_bounding_box(
                arr, x_upper_offset, x_lower_offset,
                z_upper_offset, z_lower_offset,
                y_upper_offset, y_lower_offset,
                spacing=spacing,
            )

        return SMImage(input_image.metadata, BB_array, input_image.label_array)


if __name__ == "__main__":
    tool = BoundingBox()
    asyncio.run(tool.main())
