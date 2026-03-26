import os
import uuid
import sys
import json
import shutil
import subprocess
import shlex
from typing import Iterable

from smcore import hardcore
from smcore.core import Blackboard
from smcore import agent

def random_id() -> str:
    """
    random_id returns a 8 character, random hex string
    this is useful for generating (probably) unique ids without thinking too hard about it.
    """
    return uuid.uuid4().hex[0:8]


def get_transit(bb_addr: str) -> Blackboard:
    """
    get_transit allows for slightly more clear switching of the transit layer.
    """
    return hardcore.HTTPTransit(bb_addr)
    #return hardcore.SQLiteTransit(bb_addr)

def find_folder_with_file(root_dir, target_file) -> str | None:
    for dirpath, _, filenames in os.walk(root_dir):
        if target_file in filenames:
            return dirpath  # folder that contains the file
    return None

def start_agent(runtime_dir: str, bb_addr: str, name: str, plan: dict, tool_dir: str, bb_len: int, output_dir: str, plan_id: str, out, err, data_id: str = None) -> subprocess.Popen:
    """
    start_agent handles the dispatching of agents into appropriate
    processes and working directories
    """
    # Copy the context into a working directory
    agt_work_dir = os.path.join(runtime_dir, f"{name}-{random_id()}")
    context_dir = find_folder_with_file(tool_dir, plan["code"])
    if context_dir is None:
        print(f"ERROR: run_plan.py: Code {plan['code']} cannot be found within the subfolder of {tool_dir} for {name} tool. Check you plan json.", file=sys.stderr, flush=True)
    shutil.copytree(context_dir, agt_work_dir)
    src_dir = "./smtool"
    for item in os.listdir(src_dir):
        shutil.copy2(os.path.join(src_dir, item), os.path.join(agt_work_dir, item))

    # Inject sitecustomize to keep noisy CuDNN warnings quiet inside each tool process.
    sitecustomize_path = os.path.join(agt_work_dir, "sitecustomize.py")
    prefix = "\n" if os.path.exists(sitecustomize_path) else ""
    with open(sitecustomize_path, "a", encoding="utf-8") as site_file:
        site_file.write(
            f"{prefix}import warnings\n"
            "warnings.filterwarnings(\n"
            "    'ignore',\n"
            "    message=r'.*Applied workaround for CuDNN issue.*',\n"
            "    category=UserWarning,\n"
            ")\n"
        )
        
    command = (
        f"{sys.executable} "
        f"{plan['code']} "
        f"--addr {bb_addr} "
        f"--name {name} "
        f"--config '{json.dumps(plan)}' "
        f"--bb_len {bb_len} "
        f"--plan_id {plan_id} "
        f"--output_dir {output_dir} "
    )
    if data_id is not None:
        command = command + f" --listen_tags dataset:{data_id}"
    # print(f"{command}", flush=True)
    
    env = os.environ.copy()
    warning_filter = "ignore:.*Applied workaround for CuDNN issue.*:UserWarning"
    if env.get("PYTHONWARNINGS"):
        env["PYTHONWARNINGS"] = f"{warning_filter},{env['PYTHONWARNINGS']}"
    else:
        env["PYTHONWARNINGS"] = warning_filter

    existing_ppath = env.get("PYTHONPATH")
    if existing_ppath:
        env["PYTHONPATH"] = f"{agt_work_dir}{os.pathsep}{existing_ppath}"
    else:
        env["PYTHONPATH"] = agt_work_dir

    proc = subprocess.Popen(
        shlex.split(command),
        cwd=agt_work_dir,
        stdout=out,
        stderr=err,
        env=env,
    )
        
    #return subprocess.Popen(shlex.split(command), cwd=agt_work_dir)
    return proc

async def replay_agent(
    name: str, src_bb_addr: str, dest_bb_addr: str, tags: Iterable[str]
):
    src_bb = hardcore.HTTPTransit(src_bb_addr)
    dest_bb = hardcore.HTTPTransit(dest_bb_addr)

    dest_bb.set_name(name)

    src_agt = agent.Agent(src_bb)
    dest_agt = agent.Agent(dest_bb)
    
    await src_agt.ignore_history()
    incoming_data = await src_agt.listen_for(tags)

    src_agt.start()
    print(f"replay from {src_bb_addr} -> {dest_bb_addr} on tags {tags}")
    while True:
        io_post = await incoming_data.get()
        print(f"replaying post from {src_bb_addr} -> {dest_bb_addr}")

        d = await io_post.data()
        md = await io_post.metadata()

        await dest_agt.post(md, d, io_post.tags())

# def process_check():
# # Check for any existing tool processes
#     # Build the command
#     cmd = ["pgrep", "-u", os.getenv("USER"), "-af", r"/smcore/"]

#     # Run it and capture output
#     result = subprocess.run(cmd, capture_output=True, text=True)

#     return result.stdout.strip()  # true if non-empty output
