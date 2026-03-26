"""
Tool Name: mask_logic
=================================

Description:
    Performs logical operations on one or two SMImage masks and returns the resulting mask as an SMImage.
    Supported logical operators: 'and', 'or', 'not', 'xor', 'sub', 'ifnot', 'ifor', 'incontact'.

Parameters:
    - input_1 (SMImage): First input mask (required).
    - input_2 (SMImage, optional): Second input mask, used depending on operator.
    - logical_operator (str): Logical operator to apply.
    - none_if_empty (bool, optional): If True and the result mask is empty, returns None. Default = False.

Output:
    - SMImage: The resulting mask after applying the logical operation.

Example JSON Plan:
    "mask_logic-example": {
        "code": "mask_logic.py",
        "input_1": "from some_previous_step",
        "input_2": "from another_step",
        "logical_operator": "and",
        "none_if_empty": True
    }
"""

import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage


class MaskLogicTool(SMSampleProcessor):

    async def execute(
        self,
        *,
        input_1: SMImage,
        input_2: SMImage = None,
        logical_operator: str,
        none_if_empty: bool = False,
    ) -> SMImage:

        if input_1 is None:
            return None

        # Adjust operator if 'not' and input_2 exists
        operator = logical_operator
        if input_2 is not None and logical_operator == "not":
            operator = "sub"

        arr1 = input_1.pixel_array
        arr2 = input_2.pixel_array if input_2 is not None else None

        # Ensure bitwise-safe dtypes (bool or int)
        arr1 = self.ensure_bitwise_dtype(arr1)
        if arr2 is not None:
            arr2 = self.ensure_bitwise_dtype(arr2)

        result_array = self.compute_operator(arr1, arr2, operator)

        # If none_if_empty is True and result is empty, return None
        if none_if_empty and not np.any(result_array):
            return None

        return SMImage(input_1.metadata, result_array, input_1.label_array)

    @staticmethod
    def ensure_bitwise_dtype(arr: np.ndarray) -> np.ndarray:
        """Convert array to a dtype that supports bitwise ops if needed."""
        if np.issubdtype(arr.dtype, np.bool_) or np.issubdtype(arr.dtype, np.integer):
            return arr
        return (arr != 0).astype(bool)

    @staticmethod
    def compute_operator(arr1: np.ndarray, arr2: np.ndarray, operator: str) -> np.ndarray:
        match operator:
            case "and":
                arr3 = arr1 & arr2
            case "or":
                arr3 = arr1 | arr2
            case "not":
                arr3 = ~arr1
            case "sub":
                arr3 = arr1 & (~arr2)
            case "xor":
                arr3 = arr1 ^ arr2
            case "ifnot":
                arr3 = arr1 if np.any(arr1) else arr2
            case "ifor":
                arr3 = arr1 | arr2 if np.any(arr1) and np.any(arr2) else np.zeros_like(arr1)
            case "incontact":
                contact = np.any(arr1 & arr2)
                arr3 = arr1 | arr2 if contact else np.zeros_like(arr1)
            case _:
                raise ValueError(
                    "Unsupported logical operator. Must be one of "
                    "'and', 'or', 'not', 'xor', 'sub', 'ifnot', 'ifor', 'incontact'."
                )

        # Ensure result is integer type
        return arr3.astype(arr1.dtype)


if __name__ == "__main__":
    tool = MaskLogicTool()
    asyncio.run(tool.main())
