"""
Tool Name: resize
=================================

Description:
    Resize an SMImage to either a specified `target_shape` or to match the shape of a given `target_image`.

Parameters:            
    - input_image (SMImage): Input image to be resized.
    - target_shape (tuple, optional): Desired shape (z, y, x) in NumPy order. Must match input dimensionality.
        If the shape does not match, an empty message will be posted.
    - target_image (SMImage, optional): If provided, resizes to match this image shape.
    - order (int, optional): Logging verbosity. Spline interpolation order (0 to 5). Default is 3..
    - preserve_values (bool, optional): Preserve original values in the SMImage.pixel_array. Default is False.
        Set this to True when the SMImage.pixel_array is a mask (binary or multi label).
        It will also override order to 0.
        This is automatically applied for the label_array.

Output:
    - SMImage: The resized image.
            
Example JSON Plan:
    "image_preprocessing-resize": {
        "code": "resize.py",
        "input_image": "from input_image",
        "target_shape": [1, 512, 512],
        "order": 3
    }

Notes:
    - If both `target_shape` and `target_image` are provided, `target_image` takes priority.
    - The output image will have the same type as the input image.
    - Internally uses `skimage.transform.resize`.
    - For parameter details, see: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
"""

import asyncio
import numpy as np
from skimage.transform import resize as skresize

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

class Resize(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        target_shape: tuple = None,
        target_image: SMImage = None,
        order: int = 3,
        preserve_values: bool = False,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None

        t_shape = target_shape
        if target_image is not None:
            t_shape = target_image.pixel_array.shape

        if len(t_shape) != len(input_image.pixel_array.shape):
            msg = (
                f"target_shape {t_shape} and input_image {input_image.pixel_array.shape} dimensions do not match, "
                "returning None"
            )
            self.print_error(msg, sample_id, warning=True)
            return None

        new_image = None
        orig_shape = input_image.pixel_array.shape
        new_image = Resize.smart_resize(input_image.pixel_array, t_shape, order=order, preserve_values=preserve_values)
        # new_image = skresize(input_image.pixel_array, t_shape, order=order, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
        # new_image = new_image.astype(input_image.pixel_array.dtype)

        new_mask = None
        if input_image.label_array is not None:
            orig_shape = input_image.label_array.shape
            # new_mask = skresize(input_image.label_array, t_shape, order=order, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
            # Always preserve values for the label_array
            new_mask = Resize.smart_resize(input_image.label_array, t_shape, order=0, preserve_values=True)

        new_metadata = None
        if input_image.metadata is not None:
            new_metadata = input_image.metadata.copy()
            if "spacing" in new_metadata and orig_shape is not None:
                new_metadata["spacing"] = new_metadata["spacing"][::-1] # Spacing metadata is (x,y,z), numpy array dimensions are (z,y,x)
                new_metadata["spacing"] = tuple(a*b/float(c) for a,b,c in zip(new_metadata["spacing"], orig_shape, t_shape))
                new_metadata["spacing"] = new_metadata["spacing"][::-1] # Convert back to (x,y,z) order

        self.print_log(f"new image dimensions: {new_image.shape}", sample_id)
        
        return SMImage(new_metadata, new_image, new_mask)
    
    @staticmethod
    def smart_resize(image: np.ndarray, new_shape, order: int | None = None, preserve_values: bool = False) -> np.ndarray:
        """
        Resize an image or mask intelligently.
        - preserve_values=True ensures output only contains values in input.
        """
        from skimage.transform import resize as skresize
        import numpy as np

        # If preserve_values is requested, force nearest-neighbor behavior
        if preserve_values:
            order = 0
            anti_alias = False
        else:
            if order is None:
                order = 1 if np.issubdtype(image.dtype, np.floating) else 0
            anti_alias = (order > 0)

        resized = skresize(
            image,
            new_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=anti_alias
        )

        return resized.astype(image.dtype)

if __name__ == "__main__":   
    tool = Resize()
    asyncio.run(tool.main())
