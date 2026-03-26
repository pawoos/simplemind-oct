"""
Tool Name: read_sm_image
=================================

Description:
    Reads a byte stream of array data into an SMImage.

Parameters:            
    - image_bytes (bytes): Input image array.
    - label_mask_bytes (bytes, optional): Label array.
        Expected to be int values representing each mask of the dimensions as the image_bytes.

Output:
    - SMImage.
            
Example JSON Plan:
    "read_sm_image": {
        "code": "read_sm_image.py",
        "image_bytes": "from dataset_upload image file",
        "label_mask_bytes": "from dataset_upload mask file",
        "final_output": true
    }

Notes:
    - Typically provides the final_output for an input_image_plan.
"""

import asyncio
import numpy as np
import io

import SimpleITK as sitk
import pydicom

from fake_file import FakeFile
from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

class ReadSMImage(SMSampleProcessor):

    async def execute(
        self,
        *,
        image_bytes: bytes, 
        label_mask_bytes: bytes, 
        msg_tags: list[str]
    ) -> SMImage:

        metadata, pixel_array = await self.read_image(image_bytes, msg_tags)
        if pixel_array is None:
            return None
        
        #self.print_log(f"label_mask_bytes = {label_mask_bytes}")
        _, label_mask_array = await self.read_image(label_mask_bytes, msg_tags)
        sm_image = SMImage(metadata, pixel_array, label_mask_array)
        
        # self.print_log(f"creating SMImage with dimensions={pixel_array.shape}")
        
        return sm_image
        
    
    async def read_image(self, data: bytes, tags: list[str]):
        if data is None:
            return None, None
        file_obj= io.BytesIO(data)

        metadata = {}
        image = None
        if "dicom" in tags:
            ds = pydicom.dcmread(file_obj)
            self.print_log(f"reader received a dicom file for modality {ds.Modality}")
            metadata["modality"] = ds.Modality
        elif "nifti" in tags:
            self.print_log("nifti file received")

            # FakeFile here helps us with a temporary file
            try:
                with FakeFile(file_obj, ".nii.gz") as ff:
                    ds = sitk.ReadImage(ff)

                    # TODO: extract metadata
                    metadata["spacing"] = ds.GetSpacing()
                    metadata["origin"] = ds.GetOrigin()
                    metadata["direction"] = ds.GetDirection()

                    # TODO: extract imagedata
                    image = sitk.GetArrayFromImage(ds)

            except Exception as e:
                self.print_error("nifti read failed")
                raise
            
        elif "npz" in tags:
            self.print_log("npz (numpy) file received")
            file_obj.seek(0)  # ensure start
            npz_file = np.load(file_obj, allow_pickle=True) 
            self.print_log(f"Keys: {npz_file.files}")
            # Try common fields
            if "imgs" in npz_file.files:
                image = npz_file["imgs"]
            elif "image" in npz_file.files:
                image = npz_file["image"]
            else:
                # fallback: take first array
                first_key = npz_file.files[0]
                image = npz_file[first_key]

            # Optional metadata
            if "spacing" in npz_file.files:
                metadata["spacing"] = npz_file["spacing"].tolist()
            if "origin" in npz_file.files:
                metadata["origin"] = npz_file["origin"].tolist()
            if "direction" in npz_file.files:
                metadata["direction"] = npz_file["direction"].tolist()
                        
        elif "png" in tags:
            pass  # TODO: handle png
        else:
            pass

        return metadata, image

if __name__ == "__main__":   
    tool = ReadSMImage()
    asyncio.run(tool.main())
