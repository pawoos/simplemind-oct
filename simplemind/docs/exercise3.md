# Exercise 3: Decision Trees

In this exercise, you’ll learn how SimpleMind uses a **decision tree** to post-process segmentation candidates. The decision tree applies **feature rules** (area, volume, spatial relationships, etc.) to decide whether to accept or reject a candidate.

***

## 1. Decision Tree Basics

A decision tree has two types of nodes:
* **Decision node** (internal): compares a feature value (e.g., candidate area, centroid position) to a threshold.
	* If `value <= threshold`, follow the *left* child.
	* If `value > threshold`, follow the *right* child.
* **Leaf node** (terminal): contains the output class probabilities.
	* For candidate selection, outputs are `[reject_prob, accept_prob]`.
	* Example: `[0.3, 0.7]` &rarr; 70% accept, 30% reject.

Example visualization (features F1, F2):
```
                 F1:V1,T1
                   /\
            V1=<T1/  \V1>T1
                 /    \
            F2:V2,T2  [0.8, 0.2]
                /\
         V2=<T2/  \V2>T2
              /    \
      [0.1, 0.9]  [0.7, 0.3]
```

***

## 2. Specifying a Decision Tree

Decision trees are written in **JSON** format. Key attributes:
* `name`: the feature name (see list below)
* `reference`: optional reference mask for comparison (e.g., trachea)
* `threshold`: numeric cutoff for decision nodes
* `none_value`: optional fallback if a reference mask is missing (None)
	* If not provided the left subtree path is followed
* `left` / `right`: child nodes (subtrees or leaf values)
* `value`: leaf output (class probabilities)

Example (right lung selection, using trachea as reference):

```
# right_kidney_dt.yaml
{
  "name": "volume", 
  "threshold": 40000,
  "_units": "mm3",
  "left": [1.0, 0.0],
  "right": {
    "name": "volume",
    "threshold": 300000,
    "_units": "mm3",
    "left": {
      "name": "centroid_offset_x",
      "_reference": "spine",
      "threshold": -80,
      "_units": "mm",
      "left": [1.0, 0.0],
      "right": {
        "name": "centroid_offset_x",
        "_reference": "spine",
        "threshold": 0,
        "_units": "mm",
        "left": [0.0, 1.0],
        "right": [1.0, 0.0]
      }
    }
    right: [1.0, 0.0]
  }
}
```
The tree first checks the **area** (between 11,000 and 30,000 mm²). If valid, it then checks the **centroid x-offset** relative to the trachea (it should be to the right).

***

## 3. Including in a Plan

An example of incorporating a decision tree into a plan can be found in [trachea_plan.json](../plans/cxr/trachea_plan.json)
* See the `decision_tree` tool in the **candidate_selection** chunk
* The `pydt_dict` parameter provides a path to [trachea_dt.json](../plans/cxr/trachea_dt.json)
* See also [Candidate Selection](exercise1.md#candidate_selection-chunk) in Exercise 1

***


## 4. Supported Features

The feature `name` must correspond to a function in [`feature_functions.py`](../tools/reasoning/decision_tree/feature_functions.py). 

| Feature             | Units  | Reference needed? |
| ------------------- | ------ | ----------------- |
| `area`              | mm²    | No                |
| `volume`            | mm³    | No                |
| `centroid_offset_x` | mm     | Yes               |
| `centroid_offset_y` | mm     | Yes               |
| `in_contact_with`   | 0/1    | Yes               |
| `overlap_fraction`  | \[0–1] | Yes               |

Developer notes:
* Functions return a number or boolean (True &rarr; 1, False &rarr; 0).
* Return `None` if a feature can’t be computed (e.g., missing reference mask).

***

## 5. Output Format

Example decision tree output for right kidney candidates (written to `stdout-<id>.log`):

``` 
reasoning_output: [
    {
        "name": "cand_1",
        "confidence": 1.0,
        "prediction_path": "volume 62543 > 40000 -> area 62543 <= 300000 -> centroid_offset_x -27 > -80 -> centroid_offset_x -27 <= 0 -> [0.0, 1.0]"
    },
    {
        "name": "cand_2",
        "confidence": 0.0,
        "prediction_path": "area 68654 > 40000 -> area 68654 <= 300000 -> centroid_offset_x 34 > -80 -> centroid_offset_x 34 > 0 -> [1.0, 0.0]"
    }
]
```
Each candidate includes:
* `confidence`: accept probability
* `prediction_path`: full decision path taken
In this example, the first candidate is accepted.

**Questions:**
Using the right kidney decision tree above:
* What is the volume range for acceptable candidates?
* What is the confidence for a candidate with:
	* area = 90,000 mm²
	* centroid_offset_x = -40 mm
* What about a candidate with:
	* area = 118,000 mm²
	* centroid_offset_x = 10 mm


## 6. Your Task: Selecting Candidates with Decision Trees

Now you'll use decision trees to separate the lungs into right and left.

Create `right_lung_plan.json` and `left_lung_plan.json` with a **candidate_selection** chunk, similar to [trachea_plan.json](../plans/cxr/trachea_plan.json).
* The chunk should get its inputs from `lungs`.
    - These are the candidates for the right and left lungs.
* Each lung will have its own `pydt_dict` file.
* The decision tree should identify its lung (right or left) based on its **area** and position (**centroid_offset_x**) relative to the `trachea`.
    - You will need to add a `relative_to_mask` parameter to the `decision_tree` tool plan.
    - Otherwise it will not be able to compute centroid_offset_x.
    - Set the decision thresholds based on your knowledge of the anatomical relationships.
* Also include a `save_png` tool instance to generate `right_lung_overlay.png` and `left_lung_overlay.png`.

When you run the CXR Thinking, inspect the `output` folder for:
* `decision_tree/*-decision_tree.png`
* `samples/0/*-decision_tree.log`

You can also experiment with training the right_lung decision tree to learn thresholds automatically.
* See [decision_tree_learn.py](../tools/reasoning/decision_tree/decision_tree_learn.py).
* Make sure the `max_depth` parameter is set to allow both area and position features to be checked.
    - But don't make it any larger than needed, the training algorithm tends to learn overly complicated (and not generalizable) trees.
* Run with: `--dataset_csv data/rlung_cxr_images.csv --learn right_lung-decision_tree`.
* The training results will be in `output_<id>/decision_tree`: `dt_train.png` and `dt_train.json`. 
* Your final decision tree should make sense anatomically and use either your manually set thresholds, the learned thresholds, or a combination.

**Submit**: 
1. your `plan.json`, `dt.json`, `decision_tree.png` files for the right and left lung,
2. `right_lung_overlay.png` and `left_lung_overlay.png` for the 5 example chest x-rays.
