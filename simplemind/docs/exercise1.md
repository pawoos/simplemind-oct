# Exercise 1: Plans and Tools for Segmentation Inference

In this exercise, we’ll review the example you ran to segment the **trachea** on chest x-ray images. You’ll learn how **tools** are combined into a processing pipeline using **plans** in SimpleMind.

***

## 1. Running a Plan

In the *Thinking* example, you called [`run_plan`](../run_plan.py) to process a dataset:
``` bash
python run_plan.py plans/cxr --dataset_csv data/cxr_images.csv ...
```
* Loads the `plans/cxr` plan folder
* Uploads input nifti image files as byte streams
  * `dataset_csv`provides the image file paths
  * **CSV format** is documented in the [upload_dataset](../upload_dataset.py) script
* Displays a live dashboard to monitor progress

**Question:**<br>
* In the dataset CSV format, what column headings are required?

***

## 2. What is a Plan?

A **plan** is a configuration of tools into a processing pipeline.
* Each plan is defined by a single JSON file (or a folder of JSON files).
* Each file describes how tools are connected, what parameters they use, and what outputs they produce.

In the plan folder, each **object** has its own plan file: 
* `input_image_plan.json`
* `clahe_image_plan.json`
* `trachea_plan.json`
* `lungs_plan.json`

Tools in a plan file are **encapsulated** — they can only connect to each other.<br>
But one tool can be marked as the **final output** for that object by adding:
``` json
"final_output": true
```
* This allows other objects (in other plan files) to use its output.
* It is referred to by its plan (object) name, e.g., "from input_image".

**Questions:** 
* In [`input_image_plan.json`](../plans/cxr/input_image_plan.json), which tool provides the final `input_image` (for use by other objects)?
* Looking at [`read_sm_image.py`](../tools/image_processing/read_sm_image/read_sm_image.py), what class type is output?
  * (This is the image type used by SM tools.)
***

## 3. Reviewing `clahe_image_plan.json`

Configures a pipeline of image preprocessing tools. In [`plans/cxr/clahe_image_plan.json`](../plans/cxr/clahe_image_plan.json), each tool is defined like this: 
```json
  "tool_output_name": {
    "code": "code.py",
    "chunk": "chunk_name",
    "parameter1": "value1",
    "parameter2": "value2"
  }
```
* `code` &rarr; Python script to run
* `chunk` (optional) &rarr; logical grouping of tools for readability
  * describes a processing pattern (searchable description for human or generative AI)
* **parameters** &rarr; configuration values (static or dynamic)

### `image_resize` tool instance

This is an instance of the `resize` tool, it has two parameters specified in its JSON plan configuration.

**Questions:** 
* Which input parameter is `image_resize` waiting for?
* Which object provides that input?

For required parameters and tool outputs, see comments in [`resize.py`](../tools/image_processing/resize/resize.py).
* Required tool parameters come from `setup`, `execute`, and `aggregate` methods
* All required parameters must be specified in the plan
* Optional parameters have a default value

**Questions:** 
* In [`resize.py`](../tools/image_processing/resize/resize.py), which method takes the parameter values as arguments?
* What is the tool output type (from this method)?

The `image_resize` tool will be called after the `input_image` plan generates a final result.
After resize, the *image_preprocessing* chunk includes:
* `image_norm` &rarr; normalizes image to [0, 1]
* `image_clahe` &rarr; applies contrast-limited adaptive histogram equalization

**Questions:** 
* Which tool does `image_norm` get its input from?
* What sequence of preprocessing steps does this chunk perform on the input `SMImage`?

## 4. Reviewing `trachea_plan.json`

[`trachea_plan.json`](../plans/cxr/trachea_plan.json) has three main chunks:
1. **torch_seg** &rarr; run deep neural net segmentation using pytorch
2. **mask_morphology** &rarr; refine the segmentation mask with morphological closing
3. **candidate_selection** &rarr; choose the connected component for the right lung

### `torch_seg` tool instance

Runs segmentation using the `execute` method in [`torch2d_segmentation.py`](../tools/neural_net/torch_segmentation/torch2d_segmentation.py).

**Questions:** 
* Which object provides its input image?
* Are all required `setup` and `execute` method parameters specified in the plan JSON?
* Why is it necessary to apply the resize tool to the mask after segmentation?

### `candidate_selection` chunk

Processes segmentation results:
1. [`conn_comp`](../tools/mask_processing/conn_comp/conn_comp.py) &rarr; find connected components
2. [`decision_tree`](../tools/reasoning/decision_tree/decision_tree.py) &rarr; evaluate candidates 
	* Tree is defined using a dictionary format (see [Exercise 3](exercise3.md) on decision trees)
	* A dictionary can either be specified in the JSON plan, or in an external json file ([`trachea_dt.json`](../plans/cxr/trachea_dt.json))
3. [`cand_select`](../tools/reasoning/cand_select/cand_select.py) &rarr; pick the best candidate

**Questions:** 
* What data type does [`conn_comp.py`](../tools/mask_processing/conn_comp/conn_comp.py) return?
* How are connected components represented?

***

## 5. Input Parameters

Review the [how-to](input_parameters.md) on getting input parameters using "from".

**Questions:** 

In [`input_image_plan.json`](../plans/cxr/input_image_plan.json):
* For the `save_png` tool, what type of inputs will be saved as PNG?
* Which tool can they come from?
	
***

## 6. Tool Categories

SimpleMind tools fall into major categories:
- *image_processing* &rarr; take images, output images
- *neural_net* &rarr; deep learning segmentation models
- *mask_processing* &rarr; refine or filter masks
- *spatial_inference* &rarr; reasoning with masks
- *reasoning* &rarr; decision trees, fuzzy logic

See [tool documentation](tools/README.md) for details.

***

That’s the end of Exercise 1! By now you should understand:
* How plans are structured
* How tools are defined and connected
* How outputs flow between objects