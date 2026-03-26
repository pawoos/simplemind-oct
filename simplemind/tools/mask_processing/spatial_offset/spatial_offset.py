"""
Tool Name: spatial_offset
=================================

Description:
    Creates a subregion mask around the centroid of the input mask with specified
    spatial offsets along the x, y, and z axes. Offsets can be specified in either
    voxel units or millimeters.

Parameters:            
    - input_image (SMImage): Input binary mask image.
    - x_offset_1 (float, optional): Lower x bound offset from centroid.
    - x_offset_2 (float, optional): Upper x bound offset from centroid.
    - y_offset_1 (float, optional): Lower y bound offset from centroid.
    - y_offset_2 (float, optional): Upper y bound offset from centroid.
    - z_offset_1 (float, optional): Lower z bound offset from centroid.
    - z_offset_2 (float, optional): Upper z bound offset from centroid.
    - offset_unit (str, optional): One of "mm" or "voxels" (default: "voxels").

Output:
    - SMImage: The cropped subregion mask.

Example JSON Plan:
    "spatial_offset": {
        "code": "spatial_offset.py",
        "input_image": "from bounding_box",
        "x_offset_1": -20,
        "x_offset_2": 20,
        "y_offset_1": -15,
        "y_offset_2": 15,
        "z_offset_1": -10,
        "z_offset_2": 10,
        "offset_unit": "mm"
    }
"""

import asyncio
import numpy as np
from typing import List

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from simplemind.agent.reasoning.engine.feature_functions import calculate_centroid


class SpatialOffset(SMSampleProcessor):

    @staticmethod
    def get_subregion(
        arr: np.ndarray,
        z_offset_1: float,
        z_offset_2: float,
        y_offset_1: float,
        y_offset_2: float,
        x_offset_1: float,
        x_offset_2: float,
        spacing: List[float],
        offset_unit: str = "voxels",
    ) -> np.ndarray:
        mask = np.zeros(arr.shape, dtype=arr.dtype)
        arr_points = len(np.where(arr == 1)[0])
        if arr_points > 0:
            z, y, x = calculate_centroid(arr)

            # adjust offsets depending on unit
            def to_voxels(offset: float, axis: int) -> int:
                if offset is None:
                    return None
                if offset_unit == "mm":
                    return int(offset / spacing[axis])
                return int(offset)

            z1 = to_voxels(z_offset_1, 2)
            z2 = to_voxels(z_offset_2, 2)
            y1 = to_voxels(y_offset_1, 1)
            y2 = to_voxels(y_offset_2, 1)
            x1 = to_voxels(x_offset_1, 0)
            x2 = to_voxels(x_offset_2, 0)

            # bounds
            upper_z = int(arr.shape[0] - 1) if z2 is None else int(z) + z2
            lower_z = 0 if z1 is None else int(z) + z1

            upper_y = int(arr.shape[1] - 1) if y2 is None else int(y) + y2
            lower_y = 0 if y1 is None else int(y) + y1

            upper_x = int(arr.shape[2] - 1) if x2 is None else int(x) + x2
            lower_x = 0 if x1 is None else int(x) + x1

            # Ensure valid ordering
            min_z, max_z = np.min([lower_z, upper_z]), np.max([lower_z, upper_z])
            min_y, max_y = np.min([lower_y, upper_y]), np.max([lower_y, upper_y])
            min_x, max_x = np.min([lower_x, upper_x]), np.max([lower_x, upper_x])

            # Clamp to array bounds
            min_z, max_z = max(min_z, 0), min(max_z, arr.shape[0] - 1)
            min_y, max_y = max(min_y, 0), min(max_y, arr.shape[1] - 1)
            min_x, max_x = max(min_x, 0), min(max_x, arr.shape[2] - 1)

            mask[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = 1

        return mask

    async def execute(
        self,
        *,
        input_image: SMImage,
        x_offset_1: float = None,
        x_offset_2: float = None,
        y_offset_1: float = None,
        y_offset_2: float = None,
        z_offset_1: float = None,
        z_offset_2: float = None,
        offset_unit: str = "voxels",
    ) -> SMImage:

        if input_image is None:
            return None

        arr = input_image.pixel_array
        if arr.ndim == 2:  # ensure 3D
            arr = np.expand_dims(arr, axis=0)

        spacing = input_image.metadata.get("spacing", [1.0, 1.0, 1.0])

        subregion_array = self.get_subregion(
            arr,
            z_offset_1=z_offset_1, z_offset_2=z_offset_2,
            y_offset_1=y_offset_1, y_offset_2=y_offset_2,
            x_offset_1=x_offset_1, x_offset_2=x_offset_2,
            spacing=spacing,
            offset_unit=offset_unit,
        )

        return SMImage(input_image.metadata, subregion_array, input_image.label_array)


if __name__ == "__main__":
    tool = SpatialOffset()
    asyncio.run(tool.main())
