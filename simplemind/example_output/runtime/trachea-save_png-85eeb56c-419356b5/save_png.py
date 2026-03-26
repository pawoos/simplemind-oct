"""
Tool Name: save_png
=================================

Description:
    Generate a png file from an SMImage.
        For 3D, selected 2D images are included.
        Works for images and masks as the input_image.pixel_array.
        Can display the input_image.label_array as an overlay.

Parameters:            
    - input_image (SMImage): Input image to be saved as png.
    - input_mask (SMImage, optional): Displayed as an image overlay. Default = None.
        input_mask.pixel_array is the mask.
        Supersedes input_image.label_array as the overlay.
    - output_dir (str, optional): Path to which png will be saved. Default is working-<id>/output/samples/0/.
        Current directory (.) is the pipeline working directory.
    - filename (str, optional): If provided, then this will be the name of the png file.
        Default is object_chunk_tool.png.
    - show_label (bool, optional): Display the input_image label as an overlay (if it exists). Default = True.
    - flatten_axis (int, optional): Flatten an axis if 3D. Default = None.
    - mask_slice_axis (int, optional): If an input_mask is provided, show only the slice of the mask with the most pixels. 
        0=z-axis, 1=y, 2=x.
        Default = None.
        This takes priority over flatten if both are provided.
    - invert_mask (bool, optional): If true, invert masks. Default = false.
        Invert mask images and label overlays from white to black foreground.
    - mask_color (str, optional): Default is red.
        Options: 'pink', 'green', 'torquise', 'cyan', 'blue', 'brightGreen', 'orange', 
        'red', 'yellow', 'purple', 'magenta', 'parrotGreen', 'b_green', 'yellow_green', 'light_blue'
    - mask_alpha (float, optional): 0.0 is most transparent, 1.0 is solid. Default = 0.8.
    - title (str, optional): Appears at the top of the plot. Default = None.
    - mask_none (bool, optional): Save an image even if the mask/label is None. Default = True.
        Otherwise, no image is saved if the mask/label is None.

Output:
    - None (that can be used by other tools).
            
Example JSON Plan:
    "save_png": {
        "code": "save_png.py",
        "input_image": "from any SMImage",
        "output_dir": "../output"
    }
  
Notes:
    - save_png writes a png file, but no output to the Blackboard, i.e., no output that can be used as input to other tools.
"""

import asyncio
import numpy as np
import os

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage, view_image
from sm_sample_id import SMSampleID

class SavePng(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_image: SMImage,
        input_mask: SMImage = None,
        output_dir: str = None,
        filename: str = None,
        show_label: bool = True, 
        flatten_axis: int = None,
        mask_slice_axis: int = None,
        invert_mask: bool = False,
        mask_color: str = "red",
        mask_alpha: float = 0.8,
        title: str = None,
        mask_none: bool = True,
        sample_id: SMSampleID,
        msg_tags: list[str]
    ) -> None:

        if input_image is None:
            return None

        image_array = input_image.pixel_array
        
        # if np.all(image_array <= 1):
        #     self.print_error(f"SAVE_PNG: {msg_tags}", sample_id)
        #     self.print_error(f"{image_array.shape}  {image_array.shape}  {image_array.dtype}   {np.unique(image_array)}")
        # self.print_error(f"SAVE_PNG: {msg_tags}", sample_id)

        mask_array = None
        if input_mask is not None:
            mask_array = input_mask.pixel_array
        elif show_label:
            mask_array = input_image.label_array
        
        if mask_array is None and not mask_none:
            return None

        if mask_slice_axis is not None and mask_array is not None:
            image_array, mask_array = self.extract_max_mask_slice(image_array, mask_array, mask_slice_axis)            
        if flatten_axis is not None and image_array.ndim==3:
            if 0 <= flatten_axis <= 2:
                image_array = np.max(image_array, axis=flatten_axis)
            else:
                self.print_error(f"flatten_axis = {flatten_axis}, should be in range [0, 2]", sample_id)

        if invert_mask:
            image_array = self.invert_binary_array(image_array)
            if mask_array is not None:
                mask_array = self.invert_binary_array(mask_array)
            
        out_dir = self.resolve_output_dir(output_dir, sample_id.dataset)
        fp = self.sample_output_path(out_dir, sample_id)
        if filename is not None:
            fp = os.path.join(fp, filename)
        else:
            tags = self.non_sample_tags(msg_tags)
            if 'SMImage' in tags:
                tags.remove('SMImage')   
            if 'execute' in tags:
                tags.remove('execute')   
            if 'aggregate' in tags:
                tags.remove('aggregate')  
            if 'result' in tags:
                tags.remove('result')  
                                                                            
            tags = self.move_to_start(tags, f"-{self.plan_id}")
            
            # For external listeners we repeat the tag without the plan_id for final_output tools
            # Remove this extra tag (if it's present) when forming the file name
            prefix_to_ignore = tags[0].split(f"-{self.plan_id}", 1)[0]
            tags = [t for t in tags if t != prefix_to_ignore]

            fname = "_".join(tags)
            fname = fname.replace(f"-{self.plan_id}", "")           
            fp = os.path.join(fp, fname+".png")        
        
        # self.print_error(f"{fp}", sample_id)        
        try:
            if mask_array is None:
                view_image(image_array, fp, input_image.metadata['spacing'])
            else:
                view_image(image_array, fp, input_image.metadata['spacing'], mask_array, alpha=mask_alpha,mask_name=title,mask_color=mask_color)
        except:
            if mask_array is None:
                view_image(image_array, fp)
            else:
                view_image(image_array, fp, mask=mask_array, alpha=mask_alpha,mask_name=title,mask_color=mask_color)       

    @staticmethod
    def non_sample_tags(msg_tags: list[str]) -> list[str]:
        """
        Returns a list of message tags that do not contain `dataset:`, `sample:`, or `total:`.
        """
        return [item for item in msg_tags if not item.startswith(("dataset:", "sample:", "total:"))]

    @staticmethod
    def move_to_start(strings, substring):
        for i, s in enumerate(strings):
            if substring in s:
                # Pop out the first match and insert it at the start
                strings.insert(0, strings.pop(i))
                break
        return strings
    
    @staticmethod
    def invert_binary_array(arr: np.ndarray) -> np.ndarray:
        # Check if array has only 0 and 1
        if np.isin(arr, [0, 1]).all():
            # Invert: 0 -> 1, 1 -> 0
            return 1 - arr
        else:
            return arr
        
    @staticmethod
    def extract_max_mask_slice(image_array: np.ndarray, mask_array: np.ndarray, mask_slice_axis: int):
        """
        Extracts the 2D slice (from both image_array and mask_array) along the given axis
        where the mask has the largest number of non-zero values.

        If mask is all zeros, returns the middle slice along the axis.
        If arrays are 2D, they are returned unchanged.

        Raises:
            TypeError: if mask_slice_axis is not an int.
            ValueError: if mask_slice_axis is not in [0, 1, 2].

        Returns:
            tuple[np.ndarray, np.ndarray]: (image_slice, mask_slice) — both 2D.
        """
        # --- Validate axis ---
        if not isinstance(mask_slice_axis, int):
            raise TypeError(f"mask_slice_axis must be an int, got {type(mask_slice_axis).__name__}")

        if mask_slice_axis not in (0, 1, 2):
            raise ValueError(f"mask_slice_axis must be one of [0, 1, 2], got {mask_slice_axis}")

        # --- Handle 2D case ---
        if mask_array.ndim == 2:
            return image_array, mask_array

        if mask_array.ndim != 3:
            raise ValueError(f"Expected 2D or 3D mask_array, got {mask_array.ndim}D")

        # --- Count non-zero elements per slice ---
        nonzero_counts = np.count_nonzero(mask_array, axis=tuple(i for i in range(3) if i != mask_slice_axis))
        total_nonzero = np.sum(nonzero_counts)

        if total_nonzero == 0:
            # Fallback: use the middle slice if mask is entirely empty
            best_slice_idx = mask_array.shape[mask_slice_axis] // 2
        else:
            best_slice_idx = int(np.argmax(nonzero_counts))

        # --- Extract the slices ---
        image_slice = np.take(image_array, best_slice_idx, axis=mask_slice_axis)
        mask_slice = np.take(mask_array, best_slice_idx, axis=mask_slice_axis)

        return image_slice, mask_slice



if __name__ == "__main__":   
    tool = SavePng()
    asyncio.run(tool.main())
