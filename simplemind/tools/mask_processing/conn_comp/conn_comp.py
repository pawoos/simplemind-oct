"""
Tool Name: conn_comp
=================================

Description:
    Performs connected component analysis on an SMImage mask and returns a mask as an SMImage.
    Each connected component is assigned an integer value, starting from 1 (0 is background).

Parameters:            
    - input_image (SMImage): Input image to be processed.
    - connectivity (int, optional): Only 4,8 (2D) and 26, 18, and 6 (3D) are allowed. Default = 6.
    - voxel_count_threshold (int, optional): Only include components with this number of voxels or more.
    - binary_mask (bool, optional): If true then all components (meeting voxel_count_threshold) are returned as a single binary mask.
        Default = False.

Output:
    - SMImage: The connected component mask image.
            
Example JSON Plan:
    "candidate_selection-conn_comp": {
        "code": "conn_comp.py",
        "input_image": "from mask_processing-morph_close",
        "voxel_count_threshold": 100
    }

Notes:
    - Uses the cc3d package: https://pypi.org/project/connected-components-3d/
"""

import asyncio
import cc3d

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage

class ConnComp(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        connectivity: int = 6,
        voxel_count_threshold: int = 0,
        binary_mask: bool = False
    ) -> SMImage:  

        if input_image is None:
            return None
        
        # Drop leading channel dim if present (SMImage now stores masks as [C, Z, Y, X])
        arr = input_image.pixel_array
        restore_channel = False
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected single-channel mask, got shape {arr.shape}")
            arr = arr[0]
            restore_channel = True

        if voxel_count_threshold > 0:
            arr = cc3d.dust(arr, connectivity=connectivity, threshold=voxel_count_threshold)
            
        conn_comp_arr = cc3d.connected_components(arr, connectivity=connectivity)
        # self.print_error(f"***** conn_comp_arr.dtype = {conn_comp_arr.dtype}")
        if binary_mask:
            conn_comp_arr = (conn_comp_arr != 0).astype(conn_comp_arr.dtype)

        if restore_channel:
            conn_comp_arr = conn_comp_arr[None, ...]
        
        return SMImage(input_image.metadata, conn_comp_arr, input_image.label_array)

if __name__ == "__main__":   
    tool = ConnComp()
    asyncio.run(tool.main())
