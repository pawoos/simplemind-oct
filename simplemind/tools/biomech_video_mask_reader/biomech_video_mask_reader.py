"""
Tool Name: biomech_video_mask_reader
=================================

Description:
    Reads a byte stream of csv data containing pose coordinates into SMImage masks.
    
    The csv format has 3 rows of header and the remainder are data.
        Row 2 has the object name, Row 3 indicates whether the column data is “x”, “y”, or “likelihood” for each object point.
        The first column is the z-coordinate for all points.       

Parameters:            
    - csv_bytes (bytes): Input csv stream.
    - x_dim (int): X dimension of the output mask.
    - y_dim (int): Y dimension of the output mask.
    - likelihood_threshold (float): Minimum likelihood to include a point.
        For each point with likelihood greater than or equal to the threshold, set a point in the mask to 1.
    - object_names (list[str]): List of object names to return.

Output:
    - Blackboard messages, one message for each object SMImage.
            
Example JSON Plan:
    "dataset_upload": {
        "code": "dataset_upload.py",
        "chunk": "load_image",
        "csv_path": "from arg dataset_csv",
        "post_tags": ["biomech_video_mask"]
    },

Notes:
    - The mask data is stored in the image data (pixel_array) of SMImage.
    - The coordinate values are rounded to integers.
    - This is not a typical tool, it implements a run method and posts messages to the Blackboard.
    - Most tools implement setup, execute, and/or aggregate methods and return values (they responsible for posting to the Blackboard).
    - This tool posts outputs with a "result" tag to be compatible with other SM pipeline tools.
"""

import asyncio
import time
import io
import pandas as pd

import csv
import numpy as np
from collections import defaultdict

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage

class BiomechVideoMaskReader(SMSampleProcessor):

    async def execute(
        self,
        *,
        csv_bytes: bytes,
        x_dim: int,
        y_dim: int,
        likelihood_threshold: float,
        object_names: list[str]
    ) -> dict[str, np.ndarray]:
        
        if csv_bytes is None:
            return None
        
        text_stream = io.StringIO(csv_bytes.decode("utf-8"))
        masks = self.create_masks_from_csv(text_stream, x_dim, y_dim, likelihood_threshold, object_names)
        
        return masks
    
    def create_masks_from_csv(self, csv_textstream: io.StringIO, x_dim, y_dim, likelihood_threshold, object_names):
        """
        Reads a CSV of point coordinates and creates a dictionary of 3D masks.
        
        Args:
            csv_filename (str): Path to the CSV file.
            x_dim (int): X dimension of mask.
            y_dim (int): Y dimension of mask.
            likelihood_threshold (float): Minimum likelihood to include a point.
            object_names (list[str]): List of object names to include.
        
        Returns:
            dict[str, np.ndarray]: Dictionary of 3D masks keyed by object name.
        """
        # self.print_error(f"likelihood_threshold = {likelihood_threshold}")
        # self.print_error(f"object_names = {object_names}")
        
        # Read CSV
        reader = list(csv.reader(csv_textstream))
        
        # Row 1: unused header
        header_objects = reader[1]  # Row 2: object names
        header_types = reader[2]    # Row 3: x / y / likelihood
                
        # Map object -> (x_col, y_col, likelihood_col)
        obj_cols = defaultdict(dict)
        for col, (obj, typ) in enumerate(zip(header_objects, header_types)):
            if obj in object_names:
                obj_cols[obj][typ] = col
        
        # Get max z from column 0
        z_values = [int(round(float(row[0]))) for row in reader[3:] if row[0]]
        max_z = max(z_values)
        
        # Initialize masks
        masks = {obj: np.zeros((max_z + 1, y_dim, x_dim), dtype=np.uint8)
                for obj in object_names}
                        
        # Process rows
        for row in reader[3:]:
            try:
                z = int(round(float(row[0])))  # First column is z
            except ValueError:
                continue
            
            for obj, cols in obj_cols.items():
                try:
                    x = int(round(float(row[cols["x"]])))
                    y = int(round(float(row[cols["y"]])))
                    likelihood = float(row[cols["likelihood"]])
                except (ValueError, KeyError):
                    continue  # Skip invalid entries
                
                # Apply bounds & threshold
                if (0 <= x < x_dim) and (0 <= y < y_dim) and (0 <= z < max_z + 1):
                    if likelihood >= likelihood_threshold:
                        masks[obj][z, y, x] = 1

        # for name, mask in masks.items():
        #     self.print_error(f"mask name = {name}, mask None = {mask is None}")
            
        return masks

    async def run(self):  
        kwargs = self.get_args(self.setup)     
        await self.setup(**kwargs)
        # Main execution loop: gathers inputs, deserializes them, calls `execute`, and posts the output.
        while True:                       
            # Get arguments
            kwargs, msgs, sample_id = await self.get_execute_args()
            self.check_kwargs(self.execute, kwargs)
            # Execute and post output
            await self.post_start(msgs, sample_id, "execute")  # Log message with "execute", "start" tags
            mask_dict = await self.execute(**kwargs)
 
            # for name, mask in mask_dict.items():
            #     self.print_error(f"run mask name = {name}, run mask None = {mask is None}")
 
            for name, mask in mask_dict.items(): 
                # self.print_log(f"{name}: shape={mask.shape}, count={np.count_nonzero(mask)}")
                tags =  [
                    f"{name.lower()}",
                    "SMImage", 
                    "result"
                ]
                tags.extend(sample_id.to_list())
                await self.post(
                    None,
                    SMImage(None, mask, None),
                    tags
                )

if __name__ == "__main__":   
    tool = BiomechVideoMaskReader()
    asyncio.run(tool.main())
    
    time.sleep(9000)
    tool.print_error(
        "Destructor about to be called this will call the destructor for HTTPTransit that was created in sm_agent.main, "
        "an exception is thrown"
    )
