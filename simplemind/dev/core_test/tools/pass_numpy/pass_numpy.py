import asyncio
import numpy as np

from sm_sample_processor import SMSampleProcessor

class PassNumpy(SMSampleProcessor):

    async def execute(
        self,
        *,
        array: np.ndarray,
        sleep: float = 0.0
    ) -> np.ndarray:
        """
        Accepts a numpy array, prints the array mean and its dimensions, and returns it.

        Args:
            array (np.ndarray): Input array.
            sleep (float, optional): Time to sleep in seconds before returning the addition (just for testing purposes). 
                Default is 0.

        Returns:
            np.ndarray: The input array.
            
        config.json:   
        "pass_numpy": {
            "code": "pass_numpy_tool.py",
            "context": "./pass_numpy_tool/",
            "parameters": {
                "input_image": "from read_sm_image",
                "target_shape": [1, 512, 512],
                "order": 3,
                "preserve_range": true,
                "anti_aliasing": true
            }
        },   
        """    

        if array is None:
            return None

        mean_value = array.mean()
        print(f"pass_numpy: array with dimensions={array.shape} and mean={mean_value}")
        await self.post(None, None, [f"dims={array.shape}", f"mean={mean_value}"])
                    
        await asyncio.sleep(sleep)

        return array

if __name__ == "__main__":   
    tool = PassNumpy()
    asyncio.run(tool.main())
