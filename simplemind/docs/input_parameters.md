# Input Parameters

In a plan JSON, tool parameters can reference values from other places using `"from"`:

| **Reference** | **Description** |
|------------------|-----------------|
| **from object_name** | Final output from another object. |
| **from tool_name [optional tags]** | Output from a tool within an object plan.<br>• Only the `tool_name` is required for tools with one output.<br>• Optional tags can be used to distinguish between multiple outputs. |
| **from arg arg_name** | Argument passed to `run_plan`. |
| **from sample_arg arg_name** | Argument specified in the `upload_dataset` CSV.<br>• `--arg_name argvalue` provided in the `sample_args` column of the CSV. |
| **from any [tags]** | Any tool output with matching tags.<br>• `run_plan` adds a dataset ID tag so that the tool only listens for the particular dataset. |
