# Exercise 2: Developing a SimpleMind Tool

In this exercise, you’ll learn how to create a new tool in SimpleMind. Tools are small, modular Python scripts that process data within a plan. They live in one of four subfolders depending on their function:
* `image_processing/`
* `mask_processing/`
* `neural_net/`
* `reasoning/`

The easiest way to develop a new tool is to copy an existing one and modify it. **Be user-friendly and document the tool well** — follow the standard comment header at the top of the script.

***

## 1. `SMSampleProcessor` base class

Most tools subclass `SMSampleProcessor`. Review the class and method documentation for [smtool/sm_sample_processor.py](../smtool/sm_sample_processor.py).

**Questions:**
* What are the two methods you can override in an `SMSampleProcessor` subclass?
* Which method is for one-time setup of data structures?
* Which method processes each sample, and what return types are supported?
* Which helper method writes a log message?
* Which helper method writes an output file?

***

The **`run()`** method orchestrates setup and execution. It’s usually **not** overridden.

**Questions:**
In `run()`:
* Which method is called to fetch the arguments for `execute`, awaiting inputs from other tools?
* How many times is the cycle (get inputs &rarr; execute) repeated?

***

## 2. Image Processing Examples

Take a look at [minmax_norm.py](../tools/image_processing/minmax_norm/minmax_norm.py). 

**Questions:**
* Which method and argument provides the sample input from another tool?
* What data type does this tool return for downstream tools?

For simple image operations, `minmax_norm.py` is a good starting template. Notice the standard `main` block:
```
if __name__ == "__main__":   
    tool = MinMaxNorm()
    asyncio.run(tool.main())
```
When you create your own tool, replace `MinMaxNorm` with your class name.

***

Next, check [resize.py](../tools/image_processing/resize/resize.py). 
* The `execute` method returns a new `SMImage`, with the image size (`new_image`) and metadata modified (`new_metadata`).
* If `SMImage` has a `label_array` it is also resized.
* The `sample_id` argument is not specified in the json plan, just add it if needed for helper functions like `self.print_error`. 
* Use `self.print_error` when throwing an exception to provide more information to the developer.
* [`SMImage`](../smtool/sm_image.py) is used for tool inputs/outputs.
    * Review the class documentation.
    * Numpy arrays have dimensions: `[C, Z, Y, X]`.

**Question:**
* Which `SMImage` metadata element is being modified during resizing?

***

## 3. Neural Network Example

Review [torch_segmentation.py](../tools/neural_net/torch_segmentation/torch_segmentation.py).

**Questions:**
* In which method is the model (`self.model`) initialized?
* The example plan JSON uses `resnet50` as the backbone. What are three other available CNN backbones?
* Some parameters are only needed for training (see [torch_segmentation_learn.py](../tools/neural_net/torch_segmentation/torch_segmentation_learn.py)). Can you name one?

***

## 4. Your Task: Creating an Image Processing Tool

Now you’ll build an [**unsharp masking**](https://how.dev/answers/what-is-unsharp-masking) tool to enhance edges in chest X-rays.

**Steps:**

**1.** Copy [minmax_norm.py](../tools/image_processing/minmax_norm/minmax_norm.py) &rarr; rename it `unsharp_masking`.
* Put it in an `unsharp_masking/` subfolder
* Change the class name

**2.** Update the `execute()` method:
* Compute a blurred version of the image using [`skimage.filters.gaussian`](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian) (`preserve_range=True`)
* Implement unsharp masking:
 ``` python
  diff = input_image.pixel_array - blurred_pixel_array
  clip_value = np.percentile(np.abs(diff), 90)
  clipped_diff = np.clip(diff, -clip_value, clip_value)
  new_pixel_array = input_image.pixel_array + clipped_diff*amount
```
* Accept `sigma` and `amount` as parameters (from the JSON plan).
* **When defining parameters in the `execute()` method, the type of each parameter must be specified.**

**3.** Add a plan:
* Copy `trachea_plan.json` &rarr; `unsharp_masking.json`
* Delete everything except the `minmax_norm.py` tool, then replace it with `unsharp_masking`
* Provide required parameters for `execute`

**4.** Run the plan using the *Think* example commands.
* Check `errors.log` and `output.log`
* You should get PNGs for the right lung **plus** an `unsharp_masking.png`

**5.** Check the output folder:
* `input_image.png` &rarr; original
* `unsharp_masking.png` &rarr; enhanced with sharpened edges
* Experiment with `sigma` and `amount` values to highlight rib edges, tubes, and lines

**Submit**: 
1. your `unsharp_masking.py` and `unsharp_masking.json` files,
2. short document with different `sigma` and `amount` values and their `unsharp_masking.png` images for a representative case, indicate the best pair of `sigma` and `amount` values for highlighting edges,
3. `unsharp_masking.png` (best pair of `sigma` and `amount` values) for the 5 example chest x-rays.
