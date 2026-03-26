"""
Controller: start_mind
=================================

Description:
    Live Controller: A SimpleMind controller for starting persistent tools from a plan.

Arguments:            
    - plan: Path to JSON file or folder of JSON plans (<object>_plan.json).
    - addr (optional): Blackboard address. Default: localhost:8080.
    - tool_path (optional): Path to the folder containing SM tools. Default = "tools".
    - output_dir (optional): Path to the tool output folder. Default = ""../../output" (in the working directory).
            
Example: python start_mind.py example_plan --dataset_csv example_data/cxr_images.csv --addr bb-1.heph.com:8080 > output.log 2> errors.log

Notes:
    - The example pipes stdout to output.log and stderr to errors.log.
"""

import os
import sys
import asyncio
import argparse
import warnings

# Torch may emit noisy CuDNN warnings for missing nvrtc; hide them to keep stderr clean.
warnings.filterwarnings(
    "ignore",
    message=r".*Applied workaround for CuDNN issue.*",
    category=UserWarning,
)

import json
import curses
from contextlib import redirect_stdout, redirect_stderr

from smcore.agent import Agent
from assemble_plan import assemble_plan_folder
import torch
import dashboard
from controller_utils import random_id, get_transit, start_agent, replay_agent

global bb_addr
global config_file
global subprocesses
global run_instance
global working_dir

async def mind_processing(plan_id: str, args, out, err, agt, bb_len):
    
    if os.path.isdir(args.plan): # folder of json files
        plan_dict = assemble_plan_folder(args.plan, "_plan.json", plan_id, args=args)
        plan_file = os.path.join(runtime_dir, "plan.json")
        with open(plan_file, 'w') as f: # dump the assembled json file for debugging purposes
            json.dump(plan_dict, f, indent=2)
        print(f"plan_file = {plan_file}", flush=True)
    elif os.path.exists(args.plan):
        with open(args.plan) as f: # read the json plan file
            try:
                plan_dict = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"json format error in {args.plan}", file=sys.stderr, flush=True)
                raise exc
        print(f"plan_file = {args.plan}", flush=True)
    else:
        print(f"Plan file not found: {args.plan}", file=sys.stderr, flush=True)
        raise FileNotFoundError(f"{args.plan} was not found")
        
    # Start our agents
    processes = []
    for key in plan_dict:
        await agt.post(None, None, [f"starting agent {key}"])
        processes.append(start_agent(runtime_dir, bb_addr, key, plan_dict[key], args.tool_path, bb_len, args.output_dir, plan_id, out, err))

    # Infinite loop handling any incoming messages
    task = agt.start()

    loop = asyncio.get_running_loop()
    dashboard_done = asyncio.Future()

    def inner(stdscr):
        task = loop.create_task(dashboard.live_dashboard(stdscr, args.addr, bb_len, plan_id, working_dir))
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

async def main(plan_id: str, args, out, err):
    """
    main can be thought of as the controller's main function.
    """
    global runtime_dir
    
    # Set up the working directory for the current controller
    os.makedirs(runtime_dir, exist_ok=False)

    bb = get_transit(bb_addr)
    bb.set_name("start_mind")
    agt = Agent(bb)
    bb_len = await agt.bb.len()
    print(f"bb_len: {bb_len}", flush=True)

    ingest_task = None
    outgest_task = None
    if args.io_addr is not None:
        if args.io_input_tags:
            i_tags = ["result"]
            i_tags.extend(args.io_input_tags)
            ingest_task = asyncio.create_task(
                replay_agent("ingest-agent", args.io_addr, bb_addr, i_tags)
            )
        if args.io_output_tags:
            o_tags = ["result"]
            o_tags.extend(args.io_output_tags)
            outgest_task = asyncio.create_task(
                replay_agent("outgest-agent", bb_addr, args.io_addr, o_tags)
            )
    
    mind_task = asyncio.create_task(
        mind_processing(plan_id, args, out, err, agt, bb_len)
    )

    tasks = [
        ingest_task,
        mind_task,
        outgest_task,
    ]
    # drop None values
    tasks = [t for t in tasks if t is not None]
    await asyncio.gather(*tasks)
    

if __name__ == "__main__":
    """
    our primary command line entrypoint for the application

    here we parse our inputs, and configure any global state that will 
    be reused throughout the controller run.
    """
    ap = argparse.ArgumentParser(
        prog="start_mind.py",
        description="a basic pipeline to handle starting SimpkeMind agent tools from a configuration file",
    )
    ap.add_argument(
        "plan", 
        help="path to file or folder with json configs"
    )
    ap.add_argument(
        "--addr",
        help="set the blackboard address",
        default="localhost:8080",
    )
    ap.add_argument(
        "--io_addr",
        help="set the io blackboard address",
        default=None,
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
    ap.add_argument(
        "--io_input_tags",
        nargs="+",       # accepts multiple strings
        help="List of tags",
        default=[]
    )
    ap.add_argument(
        "--io_output_tags",
        nargs="+",       # accepts multiple strings
        help="List of tags",
        default=[]
    )
                        
    args = ap.parse_args()

    bb_addr = args.addr
    # run_id = mind_id = random_id()
    plan_instance_id = random_id()

    # Each controller instance must have a unique working directory
    working_dir = os.path.join(os.getcwd(), f"working-{plan_instance_id}")
    runtime_dir = os.path.join(working_dir, "runtime")
    if os.path.exists(working_dir):
        raise FileExistsError(f"Unexpected directory exists at {working_dir}")
    else:
        os.makedirs(working_dir)
    # print(f"working directory: {runtime_dir}")

    with open(os.path.join(working_dir, "stdout.log"), "w") as out, open(os.path.join(working_dir, "stderr.log"), "w") as err:
        with redirect_stdout(out), redirect_stderr(err):
            asyncio.run(main(plan_instance_id, args, out, err))
