"""
Tool Name: morphology
=================================

Description:
    Performs morphological processing on binary masks (SMImage) in 2D or 3D.
    Masks are assumed to contain only 0 and 1 values.

Parameters:
    - input_image (SMImage): Input binary mask to be processed (2D or 3D).
    - morphological_task (str): One of "open", "close", "erode", "dilate".
    - kernel (str): Specifies the structuring element.
        * For 2D kernels: "rectangle width height", "ellipse width height"
        * For 3D kernels: "rectangle depth height width", "ellipse depth height width", "ball radius"
        * For a rectangle with range of [-x, x] use size parameter 2x+1
    - dimensionality (int, optional): 2 or 3 (default = 2).
        Determines whether to apply morphology in 2D or 3D.

Output:
    - SMImage: The processed binary mask (values 0 and 1).

Example JSON Plan:
    # 2D morphological closing with ellipse
    "mask_processing-morph_close_2d": {
        "code": "morphology.py",
        "context": "./tools/mask_processing/morphology/",
        "input_image": "from neural_net-torch_seg",
        "morphological_task": "close",
        "kernel": "ellipse 10 10",
        "dimensionality": 2
    }

    # 3D morphological closing with a ball kernel
    "mask_processing-morph_close_3d": {
        "code": "morphology.py",
        "context": "./tools/mask_processing/morphology/",
        "input_image": "from neural_net-torch_seg",
        "morphological_task": "close",
        "kernel": "ball 5",
        "dimensionality": 3
    }

Notes:
    - Input and output masks must contain only values 0 and 1.
    - Uses skimage.morphology (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html).
    - 2D supported kernels: "rectangle", "ellipse".
    - 3D supported kernels: "rectangle", "ellipse", "ball".
"""

import asyncio
import numpy as np
import skimage.morphology as morph

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage

class Morphology(SMSampleProcessor):

    async def setup(
        self,
        *,
        morphological_task: str,
        kernel: str,
        dimensionality: int = 2  # 2 or 3
    ) -> None:

        kernel_params = kernel.split()
        kernel_shape = kernel_params[0]
        kernel_size = tuple(map(int, kernel_params[1:]))
        self.dimensionality = dimensionality

        # self.print_log(f"morphological kernel: {morphological_task} {kernel_shape} {kernel_size} (dim={dimensionality})")
        attributes_verified_dict, attribute_error_msg_dict = self.verify_attributes(
            morphological_task, kernel_shape, kernel_size, dimensionality
        )

        params_ok = True
        self.kernel_mask = None
        for key in attributes_verified_dict.keys():
            if not attributes_verified_dict[key]:
                await self.log_message(f"{attribute_error_msg_dict[key]}")
                params_ok = False

        if params_ok:
            self.kernel_mask = self.get_kernel(kernel_shape, kernel_size, dimensionality)
            
        # self.print_log(f"kernel_mask: {self.kernel_mask}")

    async def execute(
        self,
        *,
        input_image: SMImage,
        morphological_task: str,
    ) -> SMImage:

        if input_image is None:
            return None

        if self.kernel_mask is None:
            self.print_error("kernel parameters invalid", warning=True)
            return None

        roi_array = input_image.pixel_array

        # Validate mask contains only 0 and 1
        unique_vals = np.unique(roi_array)
        if not np.all(np.isin(unique_vals, [0, 1])):
            self.print_error(f"Input mask contains non-binary values: {unique_vals}", warning=True)
            return None

        new_mask = self.process_mask(morphological_task, roi_array, self.kernel_mask)

        # Ensure output stays 0/1
        new_mask = (new_mask > 0).astype(np.uint8)

        return SMImage(input_image.metadata, new_mask, input_image.label_array)

    def verify_attributes(self, morphological_task, kernel_shape, kernel_size, dimensionality):
        task_attributes = ["erode", "dilate", "open", "close"]
        shape_attributes = ["rectangle", "ellipse", "ball"]

        attributes_verified_dict = {
            "task_verified": False,
            "shape_verified": False,
            "size_verified": False,
            "dim_verified": False,
        }
        attribute_error_msg_dict = {
            "task_verified": "Please assign 'erode', 'dilate', 'open', or 'close' to 'morphological_task'.",
            "shape_verified": "Please assign 'rectangle', 'ellipse', or 'ball' to 'kernel_shape'.",
            "size_verified": "Please assign appropriate kernel size(s).",
            "dim_verified": "Dimensionality must be 2 or 3.",
        }

        attributes_verified_dict["task_verified"] = morphological_task in task_attributes
        attributes_verified_dict["shape_verified"] = kernel_shape in shape_attributes
        attributes_verified_dict["dim_verified"] = dimensionality in [2, 3]

        # size check
        if dimensionality == 2 and len(kernel_size) == 2:
            attributes_verified_dict["size_verified"] = True
        elif dimensionality == 3 and (len(kernel_size) in [1, 3]):
            attributes_verified_dict["size_verified"] = True

        return attributes_verified_dict, attribute_error_msg_dict

    def get_kernel(self, kernel_shape, kernel_size, dimensionality):
        kernel = None

        if dimensionality == 2:
            height, width = kernel_size
            if kernel_shape == "rectangle":
                kernel = morph.rectangle(height, width)
            elif kernel_shape == "ellipse":
                kernel = morph.ellipse(height, width)
            else:
                raise ValueError(f"{kernel_shape} not supported in 2D")

        elif dimensionality == 3:
            if kernel_shape == "rectangle":
                if len(kernel_size) == 1:
                    kernel = morph.cube(kernel_size[0])
                else:
                    kernel = np.ones(kernel_size, dtype=bool)
            elif kernel_shape == "ellipse":
                if len(kernel_size) != 3:
                    raise ValueError("3D ellipse requires three sizes: depth, height, width")
                z, y, x = np.ogrid[:kernel_size[0], :kernel_size[1], :kernel_size[2]]
                zc, yc, xc = (np.array(kernel_size) - 1) / 2
                kernel = ((z - zc) ** 2 / (zc**2) +
                          (y - yc) ** 2 / (yc**2) +
                          (x - xc) ** 2 / (xc**2)) <= 1
            elif kernel_shape == "ball":
                radius = kernel_size[0]
                kernel = morph.ball(radius)

        return kernel.astype(bool)

    def process_mask(self, morphological_task, roi_array, kernel):
        # Handle channel dimension from 4D SMImage ([C, Z, Y, X]) -> drop channel
        restore_channel = False
        if roi_array.ndim == 4:
            if roi_array.shape[0] != 1:
                raise ValueError(f"Expected single-channel mask, got shape {roi_array.shape}")
            roi_array = roi_array[0]  # now [Z, Y, X]
            restore_channel = True

        def apply_op(arr2d_or_3d):
            # perform morphology on a single array that matches kernel dims
            if arr2d_or_3d.ndim != kernel.ndim:
                raise ValueError(
                    f"Dimensionality mismatch: input {arr2d_or_3d.ndim}D vs kernel {kernel.ndim}D"
                )
            match morphological_task:
                case "erode":
                    return morph.erosion(arr2d_or_3d, kernel)
                case "dilate":
                    return morph.dilation(arr2d_or_3d, kernel)
                case "open":
                    return morph.opening(arr2d_or_3d, kernel)
                case "close":
                    return morph.closing(arr2d_or_3d, kernel)
            raise ValueError(f"Unsupported morphological task: {morphological_task}")

        # If we are asked to operate in 2D but have a stack of slices, apply per-slice.
        if kernel.ndim == 2 and roi_array.ndim == 3:
            processed_slices = [apply_op(roi_array[z]) for z in range(roi_array.shape[0])]
            processed_roi = np.stack(processed_slices, axis=0)
        else:
            processed_roi = apply_op(roi_array)

        # restore channel dimension if needed
        if restore_channel:
            processed_roi = np.expand_dims(processed_roi, axis=0)

        # For 2D inputs originally shaped [1, H, W], maintain that shape
        if processed_roi.ndim == 2:
            processed_roi = np.expand_dims(processed_roi, axis=0)

        return processed_roi

        match morphological_task:
            case "erode":
                processed_roi = morph.erosion(roi_array, kernel)
            case "dilate":
                processed_roi = morph.dilation(roi_array, kernel)
            case "open":
                processed_roi = morph.opening(roi_array, kernel)
            case "close":
                processed_roi = morph.closing(roi_array, kernel)

        # restore singleton dimension if needed
        if squeeze_back:
            processed_roi = np.expand_dims(processed_roi, axis=0)

        return processed_roi


if __name__ == "__main__":
    tool = Morphology()
    asyncio.run(tool.main())
