# Pipeline Run Status

The dashboard provides real time status of a pipeline run.

***

## Consec Sample

| Format     | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| **-**      | tool has started                                                        |
| **S / M**  | **S** = highest consecutive sample processed, **M** = max sample number for the run |
| __S* / M__ | sample **S** is being processed by `execute` method                     |
| **a / a**  | `aggregate` method is processing/completed                                 |


***

## Stopping the Run

* Stops automatically when all tools become inactive
  * Final output &rarr; `working-<id>/stdscr.log`
* Manually stop with **"q"** or **Ctrl-C**. 
* If processes remain, [clean them up](process_cleanup.md).<br>
