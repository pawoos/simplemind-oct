"""
Controller: upload_dataset
=================================

Description:
    A Core dataset uploader.
        It accepts a csv file containing paths to files to be uploaded to the Blackboard.

    The csv format has three column headings: "image" (required), "label_mask" (required), "sample_args" (optional).
        The "image" column is required, the "label_mask" is optional (can be blank).
        Each row specifies paths to nifti image data and its label mask.
        The paths can be absolute or relative.
        If they are relative they are given relative to the directory containing the csv file.
        An example of a supported sample_arg is --upload_tags (optional, multiple allowed).
            These are included as message tags with the upload.
            Commonly used to specify the type of data being uploaded (e.g., "nifti" or "biomech_video_mask").
        An argument, `--arg_name arg_value`, can be accessed in a plan using `from sample_arg arg_name`

Arguments:            
    - dataset_csv: path to csv containing file paths of the dataset
    - addr (optional): Blackboard address. Default: localhost:8080          
    
Example: python start_mind.py biomech_video_plan --addr public-bb.jmh.lol:8080

Notes:
    - The main function accepts an optional data_id string that is used to tag the upload messages.
    - If not provided, it generates a random data_id string that it returns.
"""

import os
import uuid
import sys
import asyncio
import argparse
import zlib
import pathlib
import pandas as pd
import argparse, shlex
from typing import Callable, Tuple, Optional
import ast
import json
import struct
import numpy as np

from smcore import hardcore
from smcore.agent import Agent
from smcore.core import Blackboard

def random_id() -> str:
    """
    random_id returns a 8 character, random hex string
    this is useful for generating (probably) unique ids without thinking too hard about it.
    """
    return uuid.uuid4().hex[0:8]

def get_transit(bb_addr: str) -> Blackboard:
    """
    get_transit allows for slightly more clear switching of the transit layer.
    """
    return hardcore.HTTPTransit(bb_addr)
    #return hardcore.SQLiteTransit(bb_addr)

async def create_agent(bb_addr):
    
    bb = get_transit(bb_addr)
    bb.set_name("upload_dataset")
    agt = Agent(bb)
    bb_len = await agt.bb.len()
    # print(f"bb_len = {bb_len}", flush=True)
    agt.last_read = bb_len - 1
    
    return agt

def resolve_image_path(csv_path: str, image_path: str) -> str:
    if not os.path.isabs(image_path):
        # interpret relative to the directory of csv_path
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        image_path = os.path.join(base_dir, image_path)
    return os.path.abspath(image_path)


def read_serialize_file(upload_path: str):
    file_path = pathlib.Path(upload_path)
    if not os.path.exists(file_path):
        print(f"The specified file does not exist: {upload_path}", file=sys.stderr, flush=True)

    with open(file_path, "rb") as f:
        data = f.read()

    return data

def read_csv(csv_arg):
    if csv_arg is None:
        err_string = "ERROR: dataset_csv is required"
        print(err_string, file=sys.stderr, flush=True)
        raise ValueError(err_string)   

    csv_path = pathlib.Path(csv_arg).resolve()
    if not csv_path.exists():
        err_string = f"ERROR: {csv_path} does not exist"
        print(err_string, file=sys.stderr, flush=True)
        raise FileNotFoundError(err_string)
    if not str(csv_path).lower().endswith(".csv"):
        err_string = f"ERROR: {csv_path} is not a CSV file"
        print(err_string, file=sys.stderr, flush=True)
        raise ValueError(err_string)

    df = pd.read_csv(csv_path)
    required_columns = {'image', 'label_mask'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required_columns}")
    
    num_rows = len(df)
    
    return df, num_rows, csv_path

def file_prep(image_path: str, mask_path: str | None) -> Tuple[Optional[bytes], Optional[bytes]]:
    image_data = read_serialize_file(image_path)
    mask_data = read_serialize_file(mask_path) if mask_path else None
    image_data = zlib.compress(image_data)
    if mask_data is not None:
        mask_data = zlib.compress(mask_data)
        
    return image_data, mask_data

def parse_unknown_args(unknown_args):
    result = {}
    key = None
    values = []

    for token in unknown_args:
        if token.startswith("--"):
            # If we're starting a new key, save the previous one
            if key is not None:
                result[key] = " ".join(values)
            key = token.lstrip("-")
            values = []
        else:
            values.append(token)

    # Save the final key-value pair
    if key is not None:
        result[key] = " ".join(values)

    # print(f"parse_unknown_args -> {result}", file=sys.stderr, flush=True)
    return result

def serialize_value(value):
    serializers = {
        int: lambda v: v.to_bytes((v.bit_length() + 8) // 8 or 1, 'big', signed=True),
        float: lambda v: struct.pack('>d', v),  # Big-endian double
        str: lambda v: v.encode('utf-8'),
        list: lambda v: json.dumps(v).encode('utf-8'),
    }

    value_type = type(value)
    if value_type in serializers:
        return serializers[value_type](value)
    else:
        raise TypeError(f"Unsupported type for serialization: {value_type.__name__}")
    
async def do_upload(
    df, 
    csv_path, 
    data_id: str, 
    num_rows: int, 
    my_data_prep: Callable[[str, str | None], Tuple[Optional[bytes], Optional[bytes]]], 
    agt):
    
    sample_arg_dict = {}
    sample_arg_dict[data_id] = {}
       
    # Post each row
    print("Uploading images", end="", flush=True, file=sys.__stdout__)
    for index, row in df.iterrows():
        image_path = resolve_image_path(csv_path, row['image'])
        mask_path = row['label_mask'] if not pd.isnull(row['label_mask']) else None
        mask_path = resolve_image_path(csv_path, mask_path) if mask_path else None
        tags = []
        print(".", end="", flush=True, file=sys.__stdout__)
        
        if 'sample_args' in df.columns and not pd.isnull(row['sample_args']):
            arg_str = row['sample_args']

            # Append CSV tags if provided
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--upload_tags",
                nargs="+",
                help="Tags to attach to the upload"
            )

            # Use parse_known_args to ignore unknown args
            args, unknown_args = parser.parse_known_args(shlex.split(arg_str))
            tags.extend(args.upload_tags)

            # Post sample_args
            sample_arg_dict = parse_unknown_args(unknown_args)
            for arg, arg_data in sample_arg_dict.items():
                converted_value = ast.literal_eval(arg_data)
                # print(f"{arg} = {converted_value} ({type(converted_value)})", file=sys.stderr, flush=True)
                value_bytes = serialize_value(converted_value)
                await agt.post(
                    None,
                    value_bytes,
                    [f"sample_arg-{arg}", f"dataset:{data_id}", f"sample:{index}", f"total:{num_rows}", "result"]
                )
                    
        image_data, mask_data = my_data_prep(image_path, mask_path)

        await agt.post(
            None,
            image_data,
            tags + ["image", "file", f"dataset:{data_id}", f"sample:{index}", f"total:{num_rows}", "result"]
        )
        await agt.post(
            None,
            mask_data,
            tags + ["mask", "file", f"dataset:{data_id}", f"sample:{index}", f"total:{num_rows}", "result"]
        )

    print(" done", flush=True, file=sys.__stdout__)
    return sample_arg_dict
        
        
async def main(args, data_id: str | None = None):
    
    if data_id is None:
        data_id = random_id()
        
    agt = await create_agent(args.addr)
    
    # Read CSV
    df, num_rows, csv_path = read_csv(args.dataset_csv)
    print(f"Number of CSV rows: {num_rows}", flush=True)
        
    # Post each CSV row
    sample_arg_dict = await do_upload(df, csv_path, data_id, num_rows, file_prep, agt)
    
    return data_id, sample_arg_dict


if __name__ == "__main__":    
    ap = argparse.ArgumentParser(
        prog="upload_dataset.py",
        description="a dataset uploader",
    )
    ap.add_argument(
        "--dataset_csv",
        help="path to csv file of of file paths"
    )
    ap.add_argument(
        "--addr",
        help="set the blackboard address",
        default="localhost:8080",
    )
    args = ap.parse_args()
    
    asyncio.run(main(args))
