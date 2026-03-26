#import uuid
#import datetime
import sys
import numpy as np
import inspect
from typing import Any, Dict, Optional, Tuple, Iterable, get_type_hints, get_origin
import argparse
import json
#import asyncio
from abc import ABC, abstractmethod
import zlib
import base64
import io

from smcore import serialize, deserialize
from smcore.core import Post
from smcore import hardcore
from smcore.agent import Agent

from sm_image import SMImage
from sm_sample_id import SMSampleID
from sm_cache import SMCache
    
class SMTool(ABC):
    """
    A base class to support tools that read messages from the BB, perform processing, and write results to the BB.
    Tools have parameters that are set via a json config file.
    
    OLD DOCUMENTATION (should this move to SMSampleProcessor?)
    SMAgent is a stateful message-processing agent that receives and handles parameterized input
    messages, tracks messages by dataset/sample, deserializes and validates inputs, and posts results.
    """

    def __init__(self) -> None:
        self.parameters = None
        self.pass_tags = False
        self.message_cache = SMCache()
        self.result_cache = SMCache()

    @staticmethod
    def get_arg_type_dict(func):
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        return {
            name: type_hints.get(name, None)
            for name in sig.parameters
            if name != "self"
        }

    @staticmethod
    def _get_sample_tags(msg_tags: list[str]) -> list[str]:
        """
        Extracts sample metadata (dataset, sample, total) from a message's labels.
        Returns None if each tag is not present exactly once in the message.
        """
        dataset_tags = [tag for tag in msg_tags if tag.startswith("dataset:")]
        sample_tags = [tag for tag in msg_tags if tag.startswith("sample:")]
        total_tags = [tag for tag in msg_tags if tag.startswith("total:")]

        if len(dataset_tags) == len(sample_tags) == len(total_tags) == 1:
            return dataset_tags + sample_tags + total_tags
        else:
            return None

    @staticmethod
    def _get_sample_tag_list(recvd_msg: Any) -> list[str]:
        """
        Extracts sample metadata (dataset, sample, total) from a message's labels.
        Returns None if each tag is not present exactly once in the message.
        """
        tags = recvd_msg.tags()
        return SMTool._get_sample_tags(tags)

    @staticmethod
    def get_sample_tag_dict(recvd_msg: Any) -> dict[str, str]:
        """
        Extracts sample metadata (dataset, sample, total) from a message's labels.
        Returns None if each tag is not present exactly once in the message.
        """
        tags = recvd_msg.tags()

        # Filter and group tags by prefix
        tag_dict = {}
        for prefix in ("dataset:", "sample:", "total:"):
            matches = [t for t in tags if t.startswith(prefix)]
            if len(matches) != 1:
                return None
            key, value = matches[0].split(":", 1)
            tag_dict[key] = value

        return tag_dict


    @staticmethod
    def param_type_handling(value, expected_type = None):
        """
        Handles parameter type conversion from json file.
        """
        if value is None:
            return None
        elif isinstance(value, str) and value.startswith("from "):
            output_value = value
        elif expected_type is None:
            output_value = value
        elif expected_type is int:
            output_value = int(value)
        elif expected_type is float:
            output_value = float(value)
        elif expected_type is tuple:
            output_value = tuple(value)
        elif expected_type is bool:
            output_value = bool(value)
        else:
            output_value = value

        return output_value

    def set_parameter_values(self) -> Dict[str, Any]:
        """
        Resolves parameter values from self.attributes or their defined defaults,
        casting to numbers if applicable.
        """
        # parameters = self.__class__.parameters_old
        result: Dict[str, Any] = {}

        for param_name, param_type in self.execute_arg_defs.items():
        # for param_name in parameters:
            # print(f"set_parameter_values: {param_name},: {param_type}", flush=True)
            
            if param_name == "msg_tags":
                    self.pass_tags = True
                    continue
            value = self.parameters.get(param_name)
            if value is not None:
                result[param_name] = self.param_type_handling(value, param_type)

        return result
    
    async def set_input_channels(self, parameter_values: Dict[str, Any], listen_tags: list[str] | None) -> Dict[str, Any]:
        """
        Registers input channels for parameters sourced with "from X" tag format.
        """
        tag_list = []
        input_channels: Dict[str, Any] = {}
        for key, value in parameter_values.items():
            if isinstance(value, str) and value.startswith("from "):
                tags = value[5:]  # Remove 'from '
                #input_channels[key] = self.listen_for(*tags.split(" "))
                t_list = tags.split(" ")
                if listen_tags is not None:
                    t_list.extend(listen_tags)
                input_channels[key] = await self.agt.listen_for(t_list)
                #input_channels[key] = await self.agt.listen_for(["write_numpy"])
                await self.agt.post(None, None, [f"listen: {','.join(t_list)}"])
                print(f"[Configure] {self._name} ({key}) is listening for: {t_list}", flush=True)
                tag_list.append(t_list)
        #print(f"returning input_channels: {input_channels}", flush=True)
        # print(f"tag_list = {tag_list}", file=sys.stderr, flush=True)
        # tag_list = list(set(tag_list))
        return input_channels, tag_list

    def io_type_handling(
        self,
        value: Any,
        param_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        need_to_deserialize: bool = False
    ) -> Any:
        """
        Handles type conversion and serialization/deserialization of parameters and outputs.
        """
        if value is None:
            return None
        
        # if isinstance(value, bytes) and len(value) == 0:
        #     return None

        if param_name is None:
            # Output serialization
            if type(value) is int:
                num_bytes = (value.bit_length() + 8) // 8  # Add a bit for sign
                return value.to_bytes(num_bytes, byteorder='big', signed=True)
            elif type(value) is float:
                return struct.pack('d', value)
            elif type(value) is np.ndarray:
                return serialize.numpy(value)
            elif type(value) is dict:
                return serialize.dictionary(value)
            elif type(value) is str:
                return value.encode('utf-8')
            elif type(value) is list:
                return json.dumps(value).encode("utf-8")
            elif type(value) is bytes:
                #return serialize.bytes(value)
                return zlib.compress(value)
            elif type(value) is SMImage:
                #return serialize.sm_image(value)
                sm_image = value
                pixel_bytes = serialize.numpy(sm_image.pixel_array)
                label_bytes = serialize.compressed_numpy(sm_image.label_array) if sm_image.label_array is not None else b''
                data = {
                    'pixel_array': base64.b64encode(pixel_bytes).decode('utf-8'),
                    'label_array': base64.b64encode(label_bytes).decode('utf-8') if label_bytes else '',
                    'metadata': sm_image.metadata,
                }
                return json.dumps(data).encode('utf-8')
            else:
                raise TypeError(f"Unhandled output type: {type(value)}")
        else:
            # Input deserialization and type-checking
            if need_to_deserialize:
                try:
                    if isinstance(value, bytes) and len(value) == 0:
                        return None
                    if expected_type is int:
                        #value = deserialize.integer(value)
                        value = int.from_bytes(value, byteorder='big', signed=True)
                    elif expected_type is float:
                        #value = deserialize.floating_point(value)
                        value = struct.unpack('d', value)[0]
                    elif expected_type is np.ndarray:
                        value = deserialize.numpy(value)
                    elif expected_type is dict:
                        value = deserialize.dictionary(value)
                    elif expected_type is str:
                        value = value.decode("utf-8")
                    elif expected_type is list:
                        value = json.loads(value.decode("utf-8"))
                    elif expected_type is bytes:
                        value = zlib.decompress(value)
                    elif expected_type is SMImage:
                        byte_stream = value
                        data = json.loads(byte_stream.decode('utf-8'))
                        pixel_bytes = base64.b64decode(data['pixel_array'])
                        with io.BytesIO(pixel_bytes) as f:
                            pixel_array = np.load(f, allow_pickle=True)
                        label_bytes = base64.b64decode(data['label_array'])
                        if len(label_bytes) == 0:
                            label_array = None
                        else:
                            with io.BytesIO(label_bytes) as f:
                                label_array = np.load(f, allow_pickle=True)['array']
                        metadata = data.get('metadata', {})
                        value = SMImage(metadata, pixel_array, label_array)
                    else:
                        raise TypeError(f"Unhandled parameter type: {expected_type}")
                except Exception:
                    print(
                        f"ERROR: {self._name} {param_name} deserialization failed: make sure that the tool input is of type {expected_type}", 
                        file=sys.stderr, flush=True
                    )
                    # print(
                    #     f"ERROR: {self._name} {param_name} value = {value}", 
                    #     file=sys.stderr, flush=True
                    # )
                    raise

            # print(f"expected_type = {expected_type}, isinstance(value, list) = {isinstance(value, list)}", 
            #     file=sys.stderr, flush=True
            # )
            type_origin = get_origin(expected_type) or expected_type
            if type_origin is list and isinstance(value, list):
                pass
            elif isinstance(value, expected_type):
                pass
            elif expected_type is float and isinstance(value, int):
                pass
            else:
                print(
                    f"ERROR: In {self._name}, parameter '{param_name}' expected {expected_type}, got {type(value).__name__}", 
                    file=sys.stderr, flush=True
                )
                if expected_type is SMImage and type(value) is str and not value.startswith("from "):
                    print(
                        f"Maybe the '{param_name}' parameter should be 'from tool'.", 
                        file=sys.stderr, flush=True
                    )
                raise TypeError(
                    f"Parameter '{param_name}' expected {expected_type}, got {type(value).__name__}"
                )

            return value

    
    async def configure(self, bb_len: int, listen_tags: list[str] | None) -> None:
        """
        Initializes parameters and input channels. 
        Ignores history to process fresh messages only.
        """
        #await self.agt.ignore_history()
        print(f"[Configure] Tool: {self._name}", flush=True) 
        self.agt.last_read = bb_len - 1
        
        self.execute_arg_defs = self.get_arg_type_dict(self.execute)
        # print(f"[Configure] arg_info: {self.execute_arg_defs}", flush=True)
        self.execute_arg_values: Dict[str, Any] = self.set_parameter_values()
        print(f"[Configure] execute_arg_values: {self.execute_arg_values}", flush=True)

        self.setup_arg_defs = self.get_arg_type_dict(self.setup)

        self.input_channels, tag_list = await self.set_input_channels(self.execute_arg_values, listen_tags)
        # print(f"[Configure] input_channels: {self.input_channels}", flush=True)
        # print(f"[Configure] channel tag list: {tag_list}", flush=True) 
        self.task = self.agt.start()

    async def get_next_msg(self, param_key: str) -> Any:
        """
        Waits for and retrieves the next message for a given parameter key.
        """
        #await self.agt.post(None, None, ["pipeline-message", "halt"])
                
        #await self.agt.post(None, None, [f"get: {param_key}"])
        msg = await self.input_channels[param_key].get()
        return msg

    async def get_next_sample_msg(self, param_key: str) -> Any:
        """
        Waits for and retrieves the next message, that has sample tags and a "result" tag, for a given parameter key.
        Skips over messages without sample tags.
        """
        print(f"{self._name} listening for: {param_key}", flush=True)
        sample_tags = None
        while sample_tags is None: # Just in case there is a log message without sample tags
            msg = await self.get_next_msg(param_key)
            if "result" in msg.tags():
                sample_tags = self.get_sample_tag_dict(msg)

        print(f"{self._name} received tags: {msg.tags()}", flush=True)
        return msg

    async def post(self, metadata: Any, data: Any, tags: list[str]) -> None:
        """
        Serializes data and posts it with the agent's name and data type tags prepended.
        """
        #data_type = f"{type(data).__module__}.{type(data).__name__}"
        data_type = type(data).__name__
        serialized_data = self.io_type_handling(data)      
        serialized_metadata = self.io_type_handling(metadata)      
        t = [self._name, data_type]   
        t.extend(tags)
        await self.agt.post(
            serialized_metadata,
            serialized_data,
            t
        )


    async def _reply(self, posts: Iterable[Post], metadata: Any, data: Any, tags: list[str]) -> None:
        """
        Serializes data and posts it with the agent's name and data type tags prepended.
        """
        data_type = type(data).__name__
        serialized_data = self.io_type_handling(data)      
        serialized_metadata = self.io_type_handling(metadata)
        t = [self._name, data_type]   
        t.extend(tags)  
        # print(f"[{self._name}] _reply t: {t}", file=sys.stderr, flush=True)
        await self.agt.reply(
            posts,
            serialized_metadata,
            serialized_data,
            t
        )

    def get_args(self, func):
        kwargs: Dict[str, Any] = {}
        
        arg_type_dict = self.get_arg_type_dict(func)

        func_sig = inspect.signature(func)
        # if "torch_seg" in self._name:
        #     print(f"func_sig = {func_sig}", file=sys.stderr, flush=True)

        for key, value in self.parameters.items():
            if key in func_sig.parameters:
                # print(f"key = {key}", file=sys.stderr, flush=True)
                if isinstance(value, str) and value.startswith("from "):
                    print(f"ERROR: {self._name}: {func.__name__} parmeter {key} cannot be from another tool", file=sys.stderr, flush=True)
                    raise ValueError("Setup parameter cannot be from another tool")

                #expected_type = self.setup_arg_defs[key]
                expected_type = arg_type_dict[key]
                kwargs[key] = self.io_type_handling(value, key, expected_type, False)
                # print("ok", file=sys.stderr, flush=True)

        return kwargs          

    async def get_execute_args(self):
        kwargs: Dict[str, Any] = {}
        sample_dict: Optional[Dict[str, str]] = None
        sample_id: Optional[SMSampleID] = None
        msgs = []

        execute_sig = inspect.signature(self.execute)
        add_sample_id = ('sample_id' in execute_sig.parameters)

        for key, value in self.execute_arg_values.items():
            if key not in execute_sig.parameters:
                continue
            deserialize_required = False
            raw_value = value

            if isinstance(value, str) and value.startswith("from "):
                # First message sets the sample
                if sample_dict is None:
                    msg = await self.get_next_sample_msg(key)
                    #sample_id = self._get_sample_tag_list(msg)
                    sample_id = SMSampleID.from_tags(msg.tags())
                    sample_dict = self.get_sample_tag_dict(msg)
                    if add_sample_id:
                        kwargs['sample_id'] = sample_id
                else:
                    while not self.message_cache.is_cached(sample_dict, key):
                        msg = await self.get_next_sample_msg(key)
                        sample_tags_from_msg = self.get_sample_tag_dict(msg)
                        self.message_cache.add(msg, sample_tags_from_msg, key)
                    msg = self.message_cache.get_data(sample_dict, key)

                msgs.append(msg)
                raw_value = await msg.data()
                deserialize_required = True

            expected_type = self.execute_arg_defs[key]
            # print(f"expected_type = {expected_type}", file=sys.stderr, flush=True)
            kwargs[key] = self.io_type_handling(raw_value, key, expected_type, deserialize_required)

        if self.pass_tags:
            msg_tags = []
            for msg in msgs:
                msg_tags.extend(msg.tags())
            msg_tags = list(set(msg_tags)) # remove duplicates
            kwargs['msg_tags'] = msg_tags

        return kwargs, msgs, sample_id    

    async def post_start(self, msgs, sample_id: SMSampleID, exec_agg: str):
        # Posts a start message for an "execute" or "aggregate" method
        tags = []
        if sample_id is not None:
            tags = sample_id.to_list()
        tags.append(exec_agg)
        tags.append("start")
        if msgs:
            await self._reply(msgs, None, None, tags)
        else:
            await self.post(None, None, tags)

    async def _post_result(self, result, msgs, sample_id: SMSampleID, exec_agg: str):
        tags = []
        if sample_id is not None:
            tags = sample_id.to_list()
        tags.append(exec_agg)
        tags.append("result")
        if "final_output" in self.parameters and self.parameters["final_output"]:
            tags.append(self._name.replace(f"-{self.plan_id}","")) # Add a tag of just the tool name (without plan_id) for external listeners to live controllers
        if msgs:
            await self._reply(msgs, None, result, tags)
        else:
            await self.post(None, result, tags)
        # if self._name=='trachea-clahe':
        #     print (f"trachea-clahe post: {sample_tags}")
            
    def check_kwargs(self, func, kwargs):
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            # Skip *args, **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            
            # If the parameter has no default and is not in kwargs → missing
            if param.default is inspect.Parameter.empty and name not in kwargs:
                print(f"{self._name}: Missing required {name} parameter from the json plan", file=sys.stderr, flush=True)
                raise TypeError(f"Missing required argument: {name}")

    async def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--addr")
        parser.add_argument("--name")
        parser.add_argument("--config")
        parser.add_argument("--bb_len")
        parser.add_argument("--output_dir")
        parser.add_argument("--plan_id")
        parser.add_argument("--listen_tags", action="append", default=None, help="Add a tag (can be used multiple times)")
        parser.add_argument(
            "--sample_args",
            nargs="+",
            default=None,
            help="add sample args from csv"
        )

        args = parser.parse_args()

        agent_config = json.loads(args.config)
        agent_name = args.name
        bb_addr = args.addr
        
        bb = hardcore.HTTPTransit(bb_addr)
        #bb = hardcore.SQLiteTransit(bb_addr)
        bb.set_name(agent_name)
        self.agt = Agent(bb)
        agent_config = json.loads(args.config)

        self._name = agent_name
        self.parameters = agent_config
        self.base_output_dir = args.output_dir
        self.plan_id = args.plan_id
        await self.agt.post(None, None, ["hello", f"plan:{self.plan_id}"])
        
        await self.configure(int(args.bb_len), args.listen_tags)        
        await self.run()