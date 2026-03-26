"""
The "alice" agent generates numpy arrays at semi random intervals and posts
them to the blackboard.
"""

import asyncio
import numpy as np
import time

from sm_sample_processor import SMSampleProcessor

global bb_addr

class WriteNumpy(SMSampleProcessor):

    async def run(self):
        task = self.agt.start()
        
        print(f"WriteNumpy", flush=True)
        
        #parameters = self.parameters()
        parameters = self.parameters
        num_arrays = parameters["num_arrays"]
        array_dim = parameters["array_dim"]
        array_value = parameters["array_value"]
        dataset_id = parameters["dataset_id"]
        reverse = parameters["reverse"]

        if not reverse:
            rng = range(num_arrays)
        else:
            rng = range(num_arrays - 1, -1, -1) 

        for index in rng:
            if array_value=='s':
                numpy_array = np.full((array_dim, array_dim), index)
            else:
                numpy_array = np.full((array_dim, array_dim), int(array_value))

            await self.post(
                None,
                numpy_array,
                [
                    self.name(),
                    f"dataset:{dataset_id}", 
                    f"sample:{index}", 
                    f"total:{num_arrays}"
                ],
            )

        task.cancel()

if __name__ == "__main__":   
    tool = WriteNumpy()
    asyncio.run(tool.main())
    
    time.sleep(9000)
    print("WriteNumpy destructor about to be called")
    print("    this will call the destructor for HTTPTransit that was created in sm_agent.main")
    print("    an exception is thrown")
