"""
Controller: run_plan
=================================

Description:
    Batch Controller: A SimpleMind controller for running a pipeline.
        A pipeline run requires a plan + dataset.
        It processes each sample in the dataset before terminating.

Arguments:            
    - plan: Path to JSON file or folder of JSON plans (<object>_plan.json).
    - dataset_csv (optional): Path to a dataset csv file of input image paths. Default = None.
        See documentation in upload_dataset.py for csv format description.
    - addr (optional): Blackboard address. Default: localhost:8080.
    - learn (optional): To train a tool, specify "<object>-<tool>".
    - tool_path (optional): Path to the folder containing SM tools. Default = "tools".
    - output_dir (optional): Path to the tool output folder. Default = ""../../output" (in the working directory).
            
Example: python run_plan.py example_plan --dataset_csv example_data/cxr_images.csv --addr bb-1.heph.com:8080 > output.log 2> errors.log

Notes:
    - The example pipes stdout to output.log and stderr to errors.log.
"""

import os
import sys
import asyncio
import argparse
import json
import curses
import subprocess
from contextlib import redirect_stdout, redirect_stderr

from smcore.agent import Agent
from assemble_plan import assemble_plan_folder, assemble_plan_file
import dashboard
import upload_dataset
from controller_utils import random_id, get_transit, start_agent, find_folder_with_file

global bb_addr
global config_file
global subprocesses
global run_instance
global working_dir


def process_check():
# Check for any existing tool processes
    # Build the command
    cmd = ["pgrep", "-u", os.getenv("USER"), "-af", r"/smcore/"]

    # Run it and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.stdout.strip()  # true if non-empty output

async def main(plan_id: str, args, out, err):
    """
    main can be thought of as the controller's main function.
    """    
    global runtime_dir
    
    if process_check():
        print(f"WARNING: SM tool processes already running", file=sys.stderr, flush=True)
    
    # Set up the working directory for the current controller
    os.makedirs(runtime_dir, exist_ok=False)

    bb = get_transit(bb_addr)
    bb.set_name("run_plan")
    agt = Agent(bb)
    bb_len = await agt.bb.len()
    print(f"bb_len = {bb_len}", flush=True)
    print(f"plan_id = {plan_id}", flush=True)
    
    data_id = args.data_id
    _, sample_arg_dict = await upload_dataset.main(args, data_id)
    
    if args.learn is not None:
        print(f"training tool: {args.learn}...", flush=True)

    # Read the json plan file
    if os.path.isdir(args.plan): # folder of json files
        plan_dict = assemble_plan_folder(args.plan, "_plan.json", plan_id, data_id, args, args.learn)
        plan_file = os.path.join(runtime_dir, "plan.json")
        with open(plan_file, 'w') as f: # dump the assembled json file for debugging purposes
            json.dump(plan_dict, f, indent=2)
        print(f"plan_file = {plan_file}", flush=True)
    elif os.path.exists(args.plan):
        plan_dict = assemble_plan_file(args.plan, "_plan.json", plan_id, data_id, args, args.learn)
        plan_file = os.path.join(runtime_dir, "plan.json")
        with open(plan_file, 'w') as f: # dump the assembled json file for debugging purposes
            json.dump(plan_dict, f, indent=2)
        print(f"plan_file = {plan_file}", flush=True)
    else:
        print(f"Plan file not found: {args.plan}", file=sys.stderr, flush=True)
        raise FileNotFoundError(f"{args.plan} was not found")
            
    # Start our agents
    processes = []
    for key in plan_dict:
        await agt.post(None, None, [f"starting agent {key}"])
        processes.append(start_agent(runtime_dir, bb_addr, key, plan_dict[key], args.tool_path, bb_len, args.output_dir, plan_id, out, err, data_id))

    # Infinite loop handling any incoming messages
    task = agt.start()
        
    loop = asyncio.get_running_loop()
    dashboard_done = asyncio.Future()

    def inner(stdscr):
        # schedule run_monitor.main on the *already running* loop
        task = loop.create_task(dashboard.single_run_dashboard(stdscr, args.addr, bb_len, plan_id, data_id, working_dir, 600))
        def _done_callback(t):
            if not dashboard_done.done():
                dashboard_done.set_result(None)
        task.add_done_callback(_done_callback)
    
    curses.wrapper(inner)
    await dashboard_done

    # Stop the started agents
    for p in processes:
        p.kill()

    task.cancel()


if __name__ == "__main__":
    """
    our primary command line entrypoint for the application

    here we parse our inputs, and configure any global state that will 
    be reused throughout the controller run.
    """
    ap = argparse.ArgumentParser(
        prog="run_plan.py",
        description="a basic pipeline to handle starting SimpkeMind agent tools from a configuration file",
    )
    ap.add_argument(
        "plan", 
        help="path to file or folder with json configs"
    )
    ap.add_argument(
        "--dry-run",
        help="enumerate the agents to be started without starting them",
    )
    ap.add_argument(
        "--addr",
        help="set the blackboard address",
        default="localhost:8080",
    )
    ap.add_argument(
        "--learn",
        help="optional object name to train",
        default=None
    )
    ap.add_argument(
        "--dataset_csv",
        help="optional path to csv file of input image paths",
        default=None
    )
    ap.add_argument(
        "--tool_path",
        help="optional path to the folder containing SM tools",
        default="tools"
    )
    ap.add_argument(
        "--output_dir",
        help="optional path to the output folder",
        default="../../output" # This will be an output folder in the runtime directory
    )
                    
    args, unknown = ap.parse_known_args()
    if unknown:
        # Stash unknown args so callers can inspect or forward them if desired
        args.unknown_args = unknown
    else:
        args.unknown_args = []
    
    if args.dataset_csv is not None:
        args.dataset_csv = os.path.abspath(args.dataset_csv)

    bb_addr = args.addr
    plan_instance_id = random_id()
    data_instance_id = random_id()
    # args.output_dir = f"{args.output_dir}-{data_instance_id}"
    args.data_id = data_instance_id # just needed by data_upload tool

    # Each controller instance must have a unique working directory
    working_dir = os.path.join(os.getcwd(), f"working-{plan_instance_id}")
    runtime_dir = os.path.join(working_dir, "runtime")
    if os.path.exists(working_dir):
        raise FileExistsError(f"Unexpected directory exists at {working_dir}")
    else:
        os.makedirs(working_dir)

    with open(os.path.join(working_dir, f"stdout-{data_instance_id}.log"), "w") as out, open(os.path.join(working_dir, f"stderr-{data_instance_id}.log"), "w") as err:
        with redirect_stdout(out), redirect_stderr(err):
            asyncio.run(main(plan_instance_id, args, out, err))
