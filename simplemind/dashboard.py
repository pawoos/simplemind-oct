import argparse
import os
import asyncio
from collections import defaultdict
import datetime
from smcore import hardcore
from smcore import agent

import curses

async def single_run_dashboard(stdscr, bb_addr, bb_len, plan_id, data_id, working_dir, wait_time):
    
    table, cols, min_cols, hello_tools = initialize_dashboard(stdscr)

    try:
        channel = await setup_message_channel(bb_addr, bb_len)
        dump_name = os.path.join(working_dir, f"stdscr-{data_id}.log")
        wtime = wait_time

        while True:
            now = datetime.datetime.now()

            # Check for messages
            no_msg_after_waiting = False
            try:
                msg = await asyncio.wait_for(channel.get(), timeout=wtime) # wait for up to wait_time sec for a message
                if not plan_id in msg.source(): # This isolates the dashboard to focus on a single plan instance
                    msg = None
                    wtime = wtime - (datetime.datetime.now() - now).total_seconds()
                    if wtime<0:
                        raise asyncio.TimeoutError("no message received in time")
                else:
                    wtime = wait_time
            except asyncio.TimeoutError:
                msg = None
                no_msg_after_waiting = True

            if msg:
                process_message(msg, table, hello_tools, now)  
                  
            # Draw table
            draw_table(table, hello_tools, plan_id, cols, min_cols, stdscr)

            if no_msg_after_waiting: # we have waited wait_time sec for a message
                none_active = all(
                    not table[data_id][source_tool]["active_samples"]
                    for source_tool in table[data_id]
                )
                if none_active:
                    print("*** Dashboard Done ***", flush=True)
                    break

            # Check for quit key
            ch = stdscr.getch()
            if ch == ord("q"):
                print("*** Dashboard Quit ***", flush=True)
                break

            await asyncio.sleep(0.05)

    finally:
        close_down(stdscr, dump_name)


async def live_dashboard(stdscr, bb_addr, bb_len, plan_id, working_dir):
    
    table, cols, min_cols, hello_tools = initialize_dashboard(stdscr)
    all_tools = {}

    try:
        channel = await setup_message_channel(bb_addr, bb_len)
        dump_name = os.path.join(working_dir, f"stdscr.log")

        while True:
            now = datetime.datetime.now()

            # Check for messages
            try:
                msg = await asyncio.wait_for(channel.get(), timeout=0.05)
            except asyncio.TimeoutError:
                msg = None

            if msg:
                process_message(msg, table, hello_tools, now)  
                for k, v in hello_tools.items():
                    if k not in all_tools:
                        all_tools[k] = v
                
            remove_inactive_runs(table)
            if not table:
                hello_tools = all_tools.copy()
                  
            # Draw table
            draw_table(table, hello_tools, plan_id, cols, min_cols, stdscr)

            # Check for quit key
            ch = stdscr.getch()
            if ch == ord("q"):
                print("*** Dashboard Quit ***", flush=True)
                break

            await asyncio.sleep(0.05)

    finally:
        close_down(stdscr, dump_name)


def initialize_dashboard(stdscr):
    # Initialize curses manually
    curses.curs_set(0)      # hide cursor
    stdscr.nodelay(True)    # non-blocking input
    stdscr.clear()

    table: dict[str, dict[str, dict]] = defaultdict(dict)
    cols = [20, 40, 20, 20]
    min_cols = sum(cols)
    hello_tools: dict[str, str] = defaultdict(dict) # tools that have posted a hello message but no result messages
    
    return table, cols, min_cols, hello_tools

def close_down(stdscr, dump_name):
    dump_screen(stdscr, dump_name)
    # Restore terminal properly
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
        
async def setup_message_channel(bb_addr, bb_len):
    if bb_addr.endswith(".db") or bb_addr.endswith(".sqlite"):
        transit = hardcore.SQLiteTransit(bb_addr)
    else:
        transit = hardcore.HTTPTransit(bb_addr)
    bb = transit
    a = agent.Agent(bb)
    a.last_read = bb_len - 1
    a.start()
    channel = await a.listen_for([])  
    return channel

def remove_inactive_runs(table):
    # print("remove_inactive_runs", flush=True)
    remove_data_id = []
    for data_id, table_data_instance in table.items():
        # print(f"data_id={data_id}", flush=True)
        tools_active = False
        for tool, tool_dict in table_data_instance.items():
            # print(f"tool={tool}, active_samples={tool_dict['active_samples']}", flush=True)
            if tool_dict["active_samples"]: # not empty
                tools_active = True
                continue
            sec = (datetime.datetime.now() - tool_dict["timestamp"]).total_seconds()
            # print(f"tool={tool}, sec={sec}", flush=True)
            if sec < 600:
                tools_active = True
                continue
            # print(f"tool={tool}, tools_active={tools_active}", flush=True)
        if not tools_active:
            remove_data_id.append(data_id)
    for data_id in remove_data_id:
        del table[data_id]
    
def process_message(msg, table, hello_tools, now):
    tags = msg.tags()
    sid = {}
    is_sample = True
    for prefix in ("dataset:", "sample:", "total:"):
        matches = [t for t in tags if t.startswith(prefix)]
        if len(matches) != 1:
            is_sample = False
            continue
        key, value = matches[0].split(":", 1)
        sid[key] = value
    if is_sample:
        data_id = sid["dataset"]
        if data_id not in table:
            table[data_id] = {}
        source_tool = msg.source()
        sample_num = int(sid["sample"])
        if "aggregate" in tags:
            sample_num = -1
        if source_tool not in table[data_id]:
            table[data_id][source_tool] = {"samples": set(), "timestamp": now}
            table[data_id][source_tool]["active_samples"] = []
            hello_tools.pop(source_tool, None)
        table[data_id][source_tool]["samples"].add(sample_num)
        table[data_id][source_tool]["timestamp"] = now
        table[data_id][source_tool]["total"] = int(sid["total"])

        if "start" in tags:
            table[data_id][source_tool]["active_samples"].append(sample_num)
        elif "result" in tags:
            if sample_num in table[data_id][source_tool]["active_samples"]:
                table[data_id][source_tool]["active_samples"].remove(sample_num)
                
    elif "hello" in tags:
        if msg.source() not in hello_tools:
            hello_tools[msg.source()] = now         
    
    
def draw_table(table, hello_tools, plan_id, cols: list[int], min_cols, stdscr):
    max_rows, max_cols = stdscr.getmaxyx()
    stdscr.clear()
    if max_rows >= 2 and max_cols >= min_cols:
        header = f"{'Plan-Data':<{cols[0]}}{'Tool':<{cols[1]}}{'Consec Sample':<{cols[2]}}{'Last Update':<{cols[3]}}"
        stdscr.addstr(0, 0, header, curses.A_BOLD)
        stdscr.addstr(1, 0, "-" * (min_cols))

        row = 2
        for data_id, sub_table in table.items():
            run_id = f"{plan_id}-{data_id}"
            for tool, sdict in sub_table.items():
                if row >= max_rows:
                    break
                samples = sdict["samples"]
                ts = sdict["timestamp"]
                if -1 in samples: # aggregate method
                    if -1 in sdict["active_samples"]:
                        max_consec_str = "a*/ a"
                    else:
                        max_consec_str = "a / a"
                else:
                    n = 0
                    while n in samples:
                        n += 1
                    max_consec = n - 1
                    max_sample_num = sdict["total"]-1
                    if max_consec in sub_table[tool]["active_samples"]:
                        max_consec_str = f"{max_consec}*/ {max_sample_num}" if max_consec >= 0 else f"/{max_sample_num}"
                    else:
                        max_consec_str = f"{max_consec} / {max_sample_num}" if max_consec >= 0 else f"/{max_sample_num}"
                tool_name = tool.replace(f"-{plan_id}", "")
                # print(f"tool={tool}, plan_id={plan_id}, data_id={data_id}, tool_name={tool_name}", flush=True)
                line = f"{run_id:<{cols[0]}}{tool_name:<{cols[1]}}{max_consec_str:<{cols[2]}}{ts}"
                try:
                    stdscr.addstr(row, 0, line[:max_cols - 1])
                except curses.error:
                    pass
                row += 1
            
        # Add tools that have said hello but not processed any cases
        dash_str = "-"
        for ht in hello_tools:
            tool_name = ht.replace(f"-{plan_id}", "")
            line = f"{plan_id:<{cols[0]}}{tool_name:<{cols[1]}}{dash_str:<{cols[2]}}{hello_tools[ht]}"
            if row >= max_rows:
                break
            try:
                stdscr.addstr(row, 0, line[:max_cols - 1])
            except curses.error:
                pass
            row += 1
    stdscr.refresh()

def dump_screen(stdscr, filename):
    height, width = stdscr.getmaxyx()
    with open(filename, 'w', encoding='utf-8') as f:
        for y in range(height):
            line = ''
            for x in range(width):
                try:
                    ch = chr(stdscr.inch(y, x) & 0xFF)
                except curses.error:
                    ch = ' '
                line += ch
            f.write(line + '\n')
        
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(prog="dashboard")
#     ap.add_argument(
#         "--addr",
#         default="localhost:8080",
#         help="set the blackboard address",
#     )
#     args = ap.parse_args()

#     loop = asyncio.get_event_loop()
#     dashboard_done = asyncio.Future()

#     def inner(stdscr):
#         task = loop.create_task(main(stdscr, args.addr, plan_id="standalone"))
#         task.add_done_callback(lambda t: dashboard_done.set_result(None))

#     curses.wrapper(inner)

#     # Wait for the dashboard to finish (press q)
#     loop.run_until_complete(dashboard_done)