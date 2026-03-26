import os
import sys
import argparse
import ast
import json

def _lookup_unknown_arg(args: argparse.Namespace | None, arg_name: str):
    """
    Attempt to resolve an argument value from args.unknown_args (tokens not parsed by argparse).
    Supports patterns: --foo bar, --foo=bar, --foo (returns True).
    """
    if args is None or not hasattr(args, "unknown_args"):
        return None
    unknown = getattr(args, "unknown_args", []) or []
    flag = f"--{arg_name}"
    def _maybe_cast(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return val
        return val

    for i, tok in enumerate(unknown):
        if tok == flag:
            # Value may be in the next token
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                return _maybe_cast(unknown[i + 1])
            return True
        if tok.startswith(f"{flag}="):
            return _maybe_cast(tok.split("=", 1)[1])
    return None

def assemble_plan_folder(plan_folder: str, json_plan_suffix: str, run_id: str = "", dataset_id: str | None = None, args: argparse.Namespace | None = None, train_tool: str | None = None) -> dict:
    '''
    Assembles a folder of SimpleMind json plan files into a single dictionary (that can be written to a single json file).
        Plans to be assembled must be of the form "object_plan.json", i.e., end with "_plan.json".
    
    Arguments:
        plan_folder (str): The folder containing the json plan files.
        json_plan_suffix (str): Filename suffix, typically "_plan.json".
        See assemble_plan for other argument descriptions.

    Returns:
        dict: Assembled plan.
    '''
    
    listdir = os.listdir(plan_folder)
    object_plan_names = [f for f in listdir if f.endswith(json_plan_suffix)] # object_plan.json filenames
    if not object_plan_names:
        print(f"WARNING: No *_plan.json files found in {plan_folder}.", file=sys.stderr, flush=True)

    return assemble_plan(plan_folder, object_plan_names, run_id, dataset_id, args, train_tool, json_plan_suffix)




def assemble_plan_file(plan_file: str, json_plan_suffix: str, run_id: str = "", dataset_id: str | None = None, args: argparse.Namespace | None = None, train_tool: str | None = None) -> dict:
    '''
    Assembles a SimpleMind json plan file.
        The filename must end with "_plan.json".
    
    Arguments:
        plan_folder (str): The folder containing the json plan files.
        json_plan_suffix (str): Filename suffix, typically "_plan.json".
        See assemble_plan for other argument descriptions.

    Returns:
        dict: Assembled plan.
    '''
    
    listdir = [plan_file]
    plan_folder = os.path.dirname(plan_file)
    object_plan_names = [os.path.basename(f) for f in listdir if f.endswith(json_plan_suffix)] # object_plan.json filenames
    if not object_plan_names:
        print(f"WARNING: {plan_file} is not valid (should be *_plan.json)", file=sys.stderr, flush=True)
    
    return assemble_plan(plan_folder, object_plan_names, run_id, dataset_id, args, train_tool, json_plan_suffix)


def assemble_plan(
    plan_folder: str, 
    object_plan_names: list, 
    run_id: str, 
    dataset_id: str | None, 
    args: argparse.Namespace | None, 
    train_tool: str | None,
    json_plan_suffix: str) -> dict:
    '''
    Assembles a folder of SimpleMind json plan files into a single dictionary (that can be written to a single json file).
    
    Plans to be assembled must be of the form "object_plan.json", i.e., end with "_plan.json".
    Uniqueness of tool names in the assembled dictionary is achieved by prepending "object-" and appending "-run_id" to each tool name.
        Parameter values of the form "from tool" are updated to "from object_tool-run_id".
        If a tool has a "final_output" parameter with value true, then its tool name in the assembled dictionary is "object-run_id".
        This allows its output to be accessed by other tools using "from object" (which is only updated to "object-run_id).
    
    Arguments:
        plan_folder (str): The folder containing the json plan files.
        run_id (str): The unique ID assigned to the pipeline run.
        dataset_id (str | None, optional): The unique dataset ID.
            This is added to "from any" tags (if provided) so that they listen for a specific data set.
        args (argparse.Namespace | None, optional): Arguments provided when running the plan. Default = None.
            If provided, then parameter values of the form "from arg arg_name" are replaced with arg values.
            Otherwise they are left as is.
        train_tool (str | None, optional): The tool to be trained. Default = None.
            If provided, the assembled plan will include only ancestors of the tool to be trained and independent tools.
        json_plan_suffix (str): Filename suffix, typically "_plan.json".

    Returns:
        dict: Assembled plan.
    '''
    
    assembled_plan = {}
    ancestor_plan = {}  # Tools to be included if train_tool is specified
    rename_map = {}     # Maps the original tool name (with run_id) to the final assembled name
    t_tool = f"{train_tool}-{run_id}" if train_tool is not None else None

    object_names = [f.removesuffix(json_plan_suffix) for f in object_plan_names] # Extract "object" for the filename
    for obj_plan_name in object_plan_names:
        obj_name = obj_plan_name.removesuffix(json_plan_suffix)
        obj_plan_file = os.path.join(plan_folder, obj_plan_name)

        with open(obj_plan_file, 'r') as f:
            tools_dict = json.load(f)

        final_output_found = False
        for tool_name, parameters in tools_dict.items():
            original_tool_name = f"{obj_name}-{tool_name}-{run_id}"
            new_tool_name = original_tool_name

            updated_parameters = {}
            ancestor_exists = False

            for param_key, param_value in parameters.items():
                # print(f"Parsing: {param_key}: {param_value}", file=sys.stderr, flush=True)
                if param_key=='final_output' and isinstance(param_value, bool) and param_value:
                    if final_output_found:
                        print(f"ERROR: {obj_plan_name}: More than one object final_output defined.", file=sys.stderr, flush=True)
                        raise ValueError(f"ERROR: {obj_plan_name}: More than one object final_output defined.")
                    else:
                        new_tool_name = f"{obj_name}-{run_id}" 

                if isinstance(param_value, str) and param_value.startswith("from "):
                    tag_list = param_value.split(" ", 3)[1:]    # Elements after "from"          
                    input_tool = tag_list.pop(0) # Allow for multiple tags, assume the first one is the input tool
                    #print(f"input_tool = {input_tool}", file=sys.stderr, flush=True)
                    if input_tool=='arg':
                        if args is None:
                            updated_parameters[param_key] = param_value
                        else:
                            if len(tag_list)==0: # no arg name is provided
                                print(f"ERROR: {new_tool_name}: The {param_key} parameter does not provide an arg name after 'arg'.", file=sys.stderr, flush=True)
                                raise ValueError(f"{new_tool_name}: The {param_key} parameter does not provide an arg name after 'arg'.")
                            arg_name = tag_list.pop(0)
                            if hasattr(args, arg_name): # check if it exists
                                updated_parameters[param_key] = getattr(args, arg_name)
                            else:
                                unknown_val = _lookup_unknown_arg(args, arg_name)
                                if unknown_val is not None:
                                    updated_parameters[param_key] = unknown_val
                                else:
                                    print(f"WARNING: {new_tool_name}: For the {param_key} parameter, no {arg_name} arg was provided when calling the python script.", file=sys.stderr, flush=True)
                                    raise ValueError(f"{new_tool_name}: For the {param_key} parameter, no {arg_name} arg was provided when calling the python script.")
                    elif input_tool=='sample_arg':
                        if len(tag_list)==0: # no arg name is provided
                            print(f"ERROR: {new_tool_name}: The {param_key} parameter does not provide an arg name after 'sample_arg'.", file=sys.stderr, flush=True)
                            raise ValueError(f"{new_tool_name}: The {param_key} parameter does not provide an arg name after 'sample_arg'.")
                        else:
                            updated_parameters[param_key] = "from sample_arg-"+tag_list.pop(0)
                    elif input_tool == 'any':
                        if len(tag_list)>0: # there are additional tags
                            tlist = " ".join(tag_list)
                            if dataset_id is not None:
                                tlist = tlist + f" dataset:{dataset_id}" # used in run_plan to restrict the listener to the dataset
                            updated_parameters[param_key] = f"from {tlist}" 
                    else:
                        if input_tool in object_names:
                            updated_input_tool = f"{input_tool}-{run_id}"
                        elif input_tool in tools_dict.keys() and "final_output" in tools_dict[input_tool] and tools_dict[input_tool]['final_output']:
                            # from tool where the tool is the final_output
                            # the tool will be renamed based on the object name
                            updated_input_tool = f"{obj_name}-{run_id}"
                        elif input_tool in tools_dict.keys(): 
                            # from tool
                            # the tool will be renamed by prepending the object name and appending the run
                            updated_input_tool = f"{obj_name}-{input_tool}-{run_id}"
                        else:
                            print(f"WARNING: {new_tool_name}: In the {param_key} parameter, {input_tool} is not a valid tool name.", file=sys.stderr, flush=True)
                            updated_input_tool = f"{input_tool}-{run_id}"
                        if len(tag_list)>0: # there are additional tags
                            updated_input_tool = updated_input_tool + " " + tag_list.pop(0)
                        updated_parameters[param_key] = f"from {updated_input_tool}" 
                        ancestor_exists = True    
                                                                
                elif isinstance(param_value, str) and param_value.endswith(".json"):
                    if not os.path.exists(param_value):
                        print(f"ERROR: {obj_plan_name}: File does not exist: {param_value}", file=sys.stderr, flush=True)
                    with open(param_value, 'r') as jsf:
                        updated_parameters[param_key] = json.load(jsf)
                        
                else:
                    updated_parameters[param_key] = param_value
            
            # Track the mapping from the unmodified name (object-tool-run_id) to the final assembled name
            rename_map[original_tool_name] = new_tool_name

            if t_tool is not None:
                # Match either the original name or the possibly renamed final_output tool name
                if new_tool_name==t_tool or original_tool_name==t_tool: # need code_learn for this object
                    if "code_learn" not in updated_parameters:
                        print(f"ERROR: {t_tool} does not have a code_learn parameter.", file=sys.stderr, flush=True)
                        print(f"{updated_parameters}", file=sys.stderr, flush=True)
                        raise ValueError(f"{t_tool} does not have a code_learn parameter.")
                    else:
                        updated_parameters["code"] = updated_parameters["code_learn"]
                if not ancestor_exists:
                    ancestor_plan[new_tool_name] = updated_parameters
                    
            assembled_plan[new_tool_name] = updated_parameters
            
    if t_tool is not None:
        # If the tool was renamed (e.g., because it is final_output), resolve to its assembled name
        assembled_t_tool = rename_map.get(t_tool, t_tool)
        if assembled_t_tool not in assembled_plan:
            available = ", ".join(sorted(assembled_plan.keys()))
            print(
                "ERROR: Requested train tool "
                f"'{t_tool}' not found in assembled plan.\n"
                f"Available tools: {available}",
                file=sys.stderr,
                flush=True,
            )
            raise KeyError(assembled_t_tool)
        ancestor_plan[assembled_t_tool] = assembled_plan[assembled_t_tool]
        get_ancestor_tools(assembled_plan, assembled_t_tool, ancestor_plan)
        assembled_plan = ancestor_plan
    
    return assembled_plan


def get_ancestor_tools(config: dict, tool_name: str, ancestors: dict):
    # Iterate through the input keys to find further dependencies
    for param_value in config[tool_name].values():
        if isinstance(param_value, str) and param_value.startswith("from "):
            # print(f"{param_value}", file=sys.stderr, flush=True)
            tag_list = param_value.split(" ", 2)
            ref_tool = tag_list[1] # allow for multiple tags, assume the first one is the tool
            if ref_tool not in ancestors and ref_tool in config:
                ancestors[ref_tool] = config[ref_tool]
                get_ancestor_tools(config, ref_tool, ancestors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="assemble_plan.py",
        description="assembles multiple plan files into one",
    )
    ap.add_argument("plan_folder", help="path to folder with json configs")
    ap.add_argument(
        "--learn",
        help="optional object name to train",
        default=None
    )
    args = ap.parse_args()
    
    plan = assemble_plan_folder(args.plan_folder)
    print("*** Assembled Plan ***")
    print(json.dumps(plan, indent=4))
