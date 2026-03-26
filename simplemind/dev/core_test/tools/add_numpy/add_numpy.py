import sys
import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor
from sm_sample_id import SMSampleID

class AddNumpy(SMSampleProcessor):

    async def execute(
        self,
        *,
        array_1: np.ndarray,
        array_2: np.ndarray,
        sleep: float = 0.0, 
        sample_id: SMSampleID
        #msg_tags: list[str]
    ) -> np.ndarray:
        """
        Adds two numpy arrays.

        Args:
            array_1 (np.ndarray): Input array.
            array_2 (np.ndarray): Input array.
            sleep (float, optional): Time to sleep in seconds before returning the addition (just for testing purposes). Default is 0.

        Returns:
            np.ndarray: Summed numpy array.
            
        config.json:
        "add_numpy": {
            "code": "add_numpy_tool.py",
            "context": "./add_numpy_tool/",
            "parameters": {
                "array_1": "from write_numpy_agent_1",
                "array_2": "from write_numpy_agent_2"
            }
        }
        """  

        if array_1 is None or array_2 is None:
            self.print_log("None", sample_id)
            # print(
            #     f"{self.name()}: {SMTool.sample_tags(msg_tags)}: None", flush=True
            # )
            return None
        
        if array_1.shape!=array_2.shape:
            warning_msg = f"array_1 shape {array_1.shape} and array2 shape {array_2.shape} do not match, returning None"
            self.print_error(warning_msg, sample_id, warning=True)
            # print(
            #     f"WARNING: {self._name}: {SMTool.sample_tags(msg_tags)}: array_1 shape {array_1.shape} and array2 shape {array_2.shape} do not match, returning None", 
            #     file=sys.stderr, flush=True
            # )
            return None

        array = array_1 + array_2
        mean_value = array.mean()
        # print(
        #     f"{self.name()}: {SMTool.sample_tags(msg_tags)}: dims={array.shape}, mean={mean_value}", flush=True
        # )
        self.print_log(f"dims={array.shape}, mean={mean_value}", sample_id)
                    
        await asyncio.sleep(sleep)

        return array

if __name__ == "__main__":   
    tool = AddNumpy()
    asyncio.run(tool.main())
