"""
Tool Name: clahe
=================================

Description:
    Applies contrast limited adaptive histogram equalization (CLAHE) to an SMImage..

Parameters:            
    - input_image (SMImage): Input image to be processed.
    - nbins (int, optional): Number of gray bins for histogram (“data range”).
    - clip_limit (float, optional): Clipping limit, normalized between 0 and 1 (higher values give more contrast).

Output:
    - SMImage: The processed image.
            
Example JSON Plan:
    "clahe": {
        "code": "clahe.py",
        "context": "./tools/clahe/",
        "input_image": "from read_sm_image",
        "nbins": 256,
        "clip_limit": 0.03
    }

Notes:
    - The output image will be converted to the same type as the input image after CLAHE.
    - Internally uses `skimage.exposure.equalize_adapthist`.
    - For parameter details, see: https://scikit-image.org/docs/0.25.x/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
"""

import asyncio
import numpy as np
from skimage import exposure

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

class Clahe(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        nbins: int = None,
        clip_limit: float = None,
        sample_id: SMSampleID
    ) -> SMImage: 

        if input_image is None:
            return None
        
        min_val = input_image.pixel_array.min()
        max_val = input_image.pixel_array.max()
        
        kwargs = {}
        kwargs['image'] = input_image.pixel_array
        if nbins is not None:
            kwargs['nbins'] = nbins
        if clip_limit is not None:
            kwargs['clip_limit'] = clip_limit
        new_pixel_array = np.array(exposure.equalize_adapthist(**kwargs))
        #new_pixel_array = np.array(exposure.equalize_adapthist(input_image.pixel_array, clip_limit = self.clip_limit,  nbins = self.nbins))
        
        if (min_val != 0) or (max_val != 1):
            new_pixel_array = new_pixel_array * (max_val - min_val) + min_val
        new_pixel_array = new_pixel_array.astype(input_image.pixel_array.dtype)
        
        # self.print_error(f"new_pixel_array: {new_pixel_array.shape}", sample_id, warning=True)
        
        return SMImage(input_image.metadata, new_pixel_array, input_image.label_array)


if __name__ == "__main__":   
    tool = Clahe()
    asyncio.run(tool.main())
