# Controller

SimpleMind provides two controllers: `run_plan.py` and `start_mind.py`.

***

## Run

**run = plan + data**

`plan_id` defines a **plan instance** of activated tools
* allows for tool grouping (isolation)
* set by the controller

`data_id` defines a **dataset instance** of samples posted to the BB
* set by the data uploader

`run_id` = `plan_id`-`data_id`

***

## Run Plan (`run_plan.py`)

**Transient tools** applied to a single data instance (batch mode) - **single run**
* calls the `upload_dataset.py` script from within `run_plan.py`
* tools listen for the single `data_id`

***

## Start Mind (`start_mind.py`)

**Persistent tools** that can be applied to multiple data instances - **multiple runs**
* `start_mind.py` activates tools
* `upload_dataset.py` is called externally
