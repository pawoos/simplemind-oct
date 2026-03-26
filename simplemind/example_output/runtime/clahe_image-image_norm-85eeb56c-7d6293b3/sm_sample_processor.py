import numpy as np
import sys
import os

from sm_image import SMImage
from sm_tool import SMTool
from sm_sample_id import SMSampleID

class SMSampleProcessor(SMTool):
    '''
    To create a configurable tool for processing samples in SimpleMind, you create a subclass of SMSampleProcessor.
        As part of a pipeline, a SampleProcessor tool accepts inputs from other SampleProcessors and returns the result of processing. 
        This result can then be an input to other Sample Processors.
    You can override setup and execute methods.
        Typically you will just override the execute method which computes and returns the processing result.
        You might override the setup method if, for example, you want to set up a morphology kernel to be applied to all samples.
    A SimpleMind tool can be instantiated with specified execute arguments via a JSON config file. 
        Input parameters can be specified in the config file as coming from other tools using "from toolname".
    For examples, see add_numpy_tool.py and resize_tool.py.
    
    Methods:
        setup: Called once before calling execute and aggregate.
        execute: Method to be implemented for processing a sample.

    Helper Methods (for use as needed within execute):
        name() -> str: Name of the tool as provided in the config file.
        print_log(log_msg: str, sample_id: SMSampleID = None): Prints a log message to stdout.
        print_error(error_msg: str, sample_id: SMSampleID = None, warning=False): Prints an error message to stderr.
        dataset_output_path(output_dir: str, sample_id: SMSampleID) -> str: Returns a path for saving dataset results.
        sample_output_path(output_dir: str, sample_id: SMSampleID) -> str: Returns a path for saving sample results.
    '''

    async def setup(self, **kwargs) -> None:
        """
        This method will be run once when the tool is set up, before processing any samples.
            You must provide type hints when declaring the execute method.
            These parameters must be statically defined in the config, i.e., not be "from" another tool.
            If a parameter is optional in the config.json, just give its default value in the execute function definition.
        """
        return None
    
    async def execute(self, **kwargs) -> int | float | dict | list | bytes | np.ndarray | SMImage | None:
        """
        Executes the sample processing.
            You must provide type hints when declaring the execute method.
            If a parameter is optional in the config.json, just give its default value in the execute function definition.
        The types in the return declaration have serialization/deserialization to and from the blackboard.
            You can only return these types.
            To expand the usable types, update sm_tool.io_type_handling.
        The execute method is called for each sample.
            To access the sample id, include an argument sample_id: SMSampleID
        """
        return None
        
    def name(self) -> str:
        return self._name

    def print_log(self, log_msg: str, sample_id: SMSampleID = None):
        sample_str = f" {sample_id}" if sample_id else ""
        print(f"{self._name}{sample_str}: {log_msg}", file=sys.stdout, flush=True)

    def print_error(self, error_msg: str, sample_id: SMSampleID = None, warning=False):
        prefix = "WARNING" if warning else "ERROR"
        sample_str = f" {sample_id}" if sample_id else ""
        print(f"{prefix}: {self._name}{sample_str}: {error_msg}", file=sys.stderr, flush=True)

    def resolve_output_dir(self, output_dir: str | None, data_id: str | None) -> str: 
        bod = self.base_output_dir
        if data_id is not None:
            bod = bod + f"_{data_id}"
        if output_dir is None:
            resolved_dir = bod
        elif not os.path.isabs(output_dir):
            resolved_dir = os.path.join(bod, output_dir)
        else:
            resolved_dir = output_dir
            
        return resolved_dir

    @staticmethod
    def sample_output_path(output_dir: str, sample_id: SMSampleID) -> str: 
        """
        Returns a path to a subfolder of `output_dir` with subfolder name given by the sample number.
        """
        fp = os.path.join(output_dir, str(sample_id.sample).zfill(len(str(sample_id.total))))
        os.makedirs(fp, exist_ok=True)
        return fp

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
            result = await self.execute(**kwargs)
            await self._post_result(result, msgs, sample_id, "execute") # results are posted with "execute", "result" tags
            