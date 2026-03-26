import numpy as np
import sys
from typing import Any, Dict, Optional, Tuple, Iterable, get_type_hints
import os
import inspect

from sm_image import SMImage
from sm_sample_processor import SMSampleProcessor
#from sm_tool import SMTool
from sm_sample_id import SMSampleID

class SMSampleAggregator(SMSampleProcessor):
    '''
    To create a configurable tool that can process samples and aggregate results, you create a subclass of SMSampleAggregator.
        It is a subclass of SMSampleProcessor and inherits its methods (review for details).
        The results from the execute method are not posted to the BB for use by other tools,
        rather they are stored in a cache and passed to the aggregate method when all samples from the dataset have a result.
    You can override the setup, execute, and aggregate methods.
        The execute method will get its input from other tools, 
        it may process the inputs or just pass the inputs directly to the aggregate method.
    A SimpleMind tool can be instantiated with specified execute arguments via a JSON config file. 
        Input parameters can be specified in the config file as coming from other tools using "from toolname".
    For examples, see decision_tree.py.
    
    Methods:
        setup: Called once before calling execute and aggregate.
        execute: Method to be implemented for processing a sample.
        aggregate: Called when all samples for a dataset have a result.

    Helper Methods:
        See those available from SMSampleProcessor.
    '''

    async def aggregate(self, dataset_id: str, results: list, total: int, **kwargs) -> int | float | dict | list | bytes | np.ndarray | SMImage | None:
        """
        """
        return None

    async def run(self):  
        kwargs = self.get_args(self.setup)     
        await self.setup(**kwargs)
        # Main execution loop: gathers inputs, deserializes them, calls `execute`, and posts the output.
        while True:                       
            # Get arguments
            kwargs, msgs, sample_id = await self.get_execute_args()
            # Execute and post output
            result = await self.execute(**kwargs)
            #await self._post_result(result, msgs, sample_id)
            self.result_cache.add(result, sample_id.to_dict(), "result")
            # self.print_error(f"sample_id: {sample_id}")

            if self.result_cache.all_samples_have_data(sample_id.dataset, "result", sample_id.total):
                dataset_results = self.result_cache.get_dataset(sample_id.dataset)
                ordered_result_list = [v['result'] for k, v in sorted(dataset_results.items(), key=lambda item: int(item[0]))]
                
                # sig = inspect.signature(self.aggregate)
                # self.print_error(f"{self.aggregate.__name__}{sig}")
                kwargs = self.get_args(self.aggregate)
                kwargs['dataset_id'] = sample_id.dataset
                kwargs['results'] = ordered_result_list
                kwargs['total'] = sample_id.total
                sid = SMSampleID(sample_id.dataset, sample_id.total, sample_id.total)
                await self.post_start(msgs, sid, "aggregate")  # Log message with "aggregate", "start" tags
                agg_result = await self.aggregate(**kwargs)
                
                self.result_cache.del_dataset(sample_id.dataset)
                await self._post_result(agg_result, msgs, sid, "aggregate")  # results are posted with "aggregate", "result" tags
                    
    @staticmethod
    def dataset_output_path(output_dir: str, dataset_id: str) -> str: 
        """
        Returns a path to a subfolder of `output_dir` with subfolder name given by the dataset id.
        Creates the folder if it doesn't exist
        """
        fp = os.path.join(output_dir, dataset_id)
        os.makedirs(fp, exist_ok=True)
        return fp