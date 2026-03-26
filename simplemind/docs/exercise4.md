# Exercise 4: Training a Segmentation Tool

In this exercise, you will run an example to train a neural network for the trachea. You’ll learn how **plans** are reused for both inference and training and how **learn tools** are implemented.

Finally, you will bring together everything learned so far. You’ll design and train a neural network to segment the heart on chest X-ray images, then use a decision tree to refine outputs and select the best candidate mask.


## 1. Running a plan for learning 

**Learning** in SM uses the same plan as Thinking but adds the `--learn` flag to specify which tool should be trained.

Example: run training of a trachea segmentation model on 300 cases.

``` bash
python run_plan.py plans/cxr --dataset_csv data/trachea_cxr_images.csv --learn trachea-torch_seg --gpu_num 0 --addr 127.0.0.1:8080
```

GPU number:
* `--gpu_num` arg is setting it to 0
* Set based on GPU availability/usage (using `nvidia-smi`)

Training samples:
* `run_plan.py` only runs upstream tools needed to generate inputs for the training tool
* See pngs in `working_<id1>/output_<id2>/samples/`

Training outputs:
* Saved in `working_<id1>/output_<id2>/trachea-torch_seg/`
* `log_training.txt` tracks performance (loss) metrics per epoch
* `test_samples` contains sample results from the test set
* Weights are saved as `checkpoint.pth`

**Question:**
* After training, what was the validation loss (mean IoU) reported in `log_training.txt`?

***

## 2. Implementing learning in a plan

A **single plan configuration** supports both inference and training.
* A `code_learn` parameter is provided for the training script **[refer to section here]**.
* Ensure all parameters needed by `setup`, `execute`, and `aggregate` methods are included.

**Questions:**
* In [trachea_plan.json](../plans/cxr/trachea_plan.json), which tool instance supports both inference and training?
* Name two parameters required only for training.
* Which parameter specifies the path to neural network weights used for inference?

***

## 3. Create the heart plan

In `plans/cxr`, start by copying `trachea.json` &rarr; `heart_plan.json`.

Update the tool configurations:
* Change the neural network `"weights_path"`
* Replace the decision tree `"pydt_dict"`
	* For now, just copy `trachea_dt.json` &rarr; `heart_dt.json`

***

## 4. Train the neural net

To set and tune the learning hyperparameters, follow the [how-to guidance](tune_pytorch.md).
* Start by deleting all of the learning parameters plan except those described in [1. Baseline Setup](tune_pytorch.md#1-baseline-setup), i.e., delete parmeters after `early_stop_patience`.
* Initialize the remaining parameters as described in the Baseline Setup.
	- The learning hyper parameters from the trachea plan may not be optimal for the heart.

Run the plan for learning similar to Step 1, with the appropriate `--learn` argument of `<plan_name>-<tool_name>`. 
* Initially use a **small dataset of 40 cases for debugging**: `--dataset_csv data/heart_cxr_images_small.csv`.
* Check `stderr-<id>.log`, `log_training.txt`, and the `samples` folder to confirm that PNGs are created for each case.
* Then, train on the **full dataset of 200 cases**: `--dataset_csv data/heart_cxr_images.csv`.

**Snapshot #1**: save `heart_plan.json`, `log_training.txt` and the `test_samples` png image with median mIoU.
* This is your baseline model performance.

***

## 5. Tune the learning hyper parameters

Follow the tuning steps in the [how-to guidance](tune_pytorch.md). 
* For each step, generate a snapshot of the best performance.
* If the performance does not improve, keep the parameters from the previous step and move to the next.

Save:
* **Snapshot #2**: After learning rate tuning.
* **Snapshot #3**: After tuning optimizer & weight decay.
* **Snapshot #4**: After encoder unfreezing.
* **Snapshot #5**: After tuning batch size.
* **Snapshot #6**: After augmentation.

***

## 6. Using the learned weights

To use the trained model for inference:
* Copy `checkpoint.pth` &rarr; `weights/cxr/heart.pth` (folder is excluded from git commits)
* If you run the plan, this should generate correct `torch_seg` tool output, but we still need to adjust the post-processing tools

***

## 7. Mask post-processing

Consider whether morphological opening or closing makes the most sense for the heart mask outputs.
* Do you need to fill or smooth the masks, both?

Update the decision tree: `heart_dt.json`
* Even though the CNN segmentation may work well for our small number of test cases, it's good to put in sanity checks in case it fails.
* At minimum, use an **area** feature
* For better robustness, check the heart position relative the trachea:
	* Add `"relative_to_mask": "from trachea"`
	* Add `centroid_offset_x` and `centroid_offset_y` features (see [feature_functions.py](../tools/reasoning/decision_tree/feature_functions.py))

***

## 8. Running and refining the plan

You can run the plan on the 5 cases from the Think example.

You can also run the plan for inference on the small training dataset: `--dataset_csv data/heart_cxr_images_small.csv` (without the `--learn` argument)
* Check the heart overlay pngs.
* If necessary, review decision tree paths for each candidate (as in [Exercise 3](exercise3.md))
* If necessary, adjust thresholds until the correct heart candidate is consistently selected

***

## 9. Submit

Submit each of the Snapshots. Indicate which Snapshot gives the best performance and, for that snapshot, also provide `heart_dt.json` and `heart.pth` files.

***

## Implementing a Learn Tool

For tools like [torch_segmentation](../tools/neural_net/torch_segmentation), we provide:
* **Inference tool** &rarr; [`torch_segmentation.py`](../tools/neural_net/torch_segmentation/torch_segmentation.py)
* **Training tool** &rarr; [`torch_segmentation_learn.py`](../tools/neural_net/torch_segmentation/torch_segmentation_learn.py)

**Key conventions:**
* Training tool filenames append `_learn`
* Training tool classes append `Learn`
* Training tools subclass [`SMSampleAggregator`](https://gitlab.com/hoffman-lab/sm-incubator/-/blob/mbrown/sm_pipeline/smtool/sm_sample_aggregator.py?ref_type=heads)

**How it works:**
* `execute` runs per sample and results are cached for training (not posted on the BB)
* once all samples are processed, `aggregate` is called with a list of results
* `aggregate` performs training (e.g., using PyTorch) and writes **weights** and **logs**
	* It does not return any output for other tools

**Questions:**
* What type does the `execute` method in [`torch_segmentation_learn.py`](../tools/neural_net/torch_segmentation/torch_segmentation_learn.py) return?
* What type is the `results` argument received by the `aggregate` method?

***

That’s the end of Exercise 4! At this point, you have:
* A trained neural net for **heart segmentation**
* A decision tree to select the best candidate
* A complete working plan (`heart_plan.json`)
