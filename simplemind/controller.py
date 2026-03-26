import os
import uuid
import sys
import asyncio
import argparse
import yaml
import json
import shutil
import subprocess
import shlex
import time

from smcore import hardcore
from smcore.agent import Agent
from smcore.core import Blackboard

global bb_addr
global config_file
global subprocesses
global run_instance
global controller_work_dir


def random_id() -> str:
    """
    random_id returns a 16 character, random hex string
    this is useful for generating (probably) unique ids without thinking too hard about it.
    """
    return uuid.uuid4().hex[0:16]


def get_transit(bb_addr: str) -> Blackboard:
    """
    get_transit allows for slightly more clear switching of the transit layer.
    """
    return hardcore.HTTPTransit(bb_addr)
    # return hardcore.SQLiteTransit(bb_addr)


def start_agent(name: str, config: dict) -> subprocess.Popen:
    """
    start_agent handles the dispatching of agents into appropriate
    processes and working directories
    """
    global controller_work_dir

    print("staring agent: ", name, "with config", config)

    # Copy the context into a working directory
    agt_work_dir = os.path.join(controller_work_dir, f"{name}-{random_id()}")
    shutil.copytree(config["context"], agt_work_dir)

    command = f"{sys.executable} {config['code']} --addr {bb_addr} --name {name} --config '{json.dumps(config)}'"
    print(command)
    return subprocess.Popen(shlex.split(command), cwd=agt_work_dir)


async def main():
    """
    main can be thought of as the controller's main function.

    all other functions are largely just to clean this up to focus
    on blackboard io behaviors.
    """
    # Set up the working directory for the current controller
    os.makedirs(controller_work_dir, exist_ok=False)

    bb = get_transit(bb_addr)
    bb.set_name("controller")
    agt = Agent(bb)

    # Say hello
    await agt.post(None, None, ["controller-hello", "hello"])

    # Channel for agents to communicate back to the controller
    controller_ch = await agt.listen_for(["controller-message"])

    # Load the configuration file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Start our agents
    processes = []
    for key in config_data:
        if key == "controller":
            continue

        await agt.post(None, None, [f"starting agent {key}"])
        processes.append(start_agent(key, config_data[key]))

    # Infinite loop handling any incoming messages
    task = agt.start()
    while True:
        print("entered main control loop")
        post = await controller_ch.get()

        if "halt" in post.tags():
            shutdown_delay = config_data["controller"]["shutdown-delay"]
            print(f"contoller HALTING in {shutdown_delay}")
            await asyncio.sleep(shutdown_delay)
            break

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
        prog="controller.py",
        description="a basic controller to handle starting simple mind agents from a configuration file",
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

    ap.add_argument("config", help="path to yaml configuration file")
    args = ap.parse_args()

    bb_addr = args.addr
    config_file = args.config
    run_instance = random_id()

    # Each controller instance must have a unique working directory
    controller_work_dir = os.path.join(
        os.getcwd(),
        f"controller_{round(time.time())}",
    )

    asyncio.run(main())
