import os
import numpy as np
import inspect
import json
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from sm_image import SMImage
import feature_functions
      
#def setup(dt_json_path: str) -> None:
def setup(pydt_dict: dict) -> None:
    """
    Initializes the decision tree.
    """  
            
    # if not os.path.exists(dt_json_path):
    #     raise RuntimeError(
    #         f"DT json file not found at: {dt_json_path}"
    #     )
    # with open(dt_json_path) as f:
    #     pydt_dict = json.load(f)

    return build_tree(pydt_dict)


def build_tree(tree_dict):
    """
    Recursively builds a decision tree from a dictionary representation.
    """
    #if "value" in tree_dict:  # Leaf node
    #    return self.DecisionTreeNode(value=tree_dict["value"]), []
    if isinstance(tree_dict, list):  # Leaf node
        return DecisionTreeNode(value=tree_dict), []
    
    left_tree, left_feature_name_list = build_tree(tree_dict["left"])
    right_tree, right_feature_name_list = build_tree(tree_dict["right"])

    none_value = tree_dict["none_value"] if 'none_value' in tree_dict else None
    
    # Create internal node
    if 'reference' in tree_dict and tree_dict["reference"] is not None:
        node = DecisionTreeNode(
            name=tree_dict["name"],
            reference=tree_dict["reference"],
            threshold=tree_dict["threshold"],
            left=left_tree,
            right=right_tree,
            none_value=none_value
        )
    else:
        node = DecisionTreeNode(
            name=tree_dict["name"],
            threshold=tree_dict["threshold"],
            left=left_tree,
            right=right_tree,
            none_value=none_value
        )            
    feature_name_item = (tree_dict["name"], tree_dict["reference"] if 'reference' in tree_dict else None)
    feature_name_list = list(set(left_feature_name_list + right_feature_name_list + [feature_name_item]))
    return node, feature_name_list

class DecisionTreeNode:
    def __init__(self, name=None, reference=None, threshold=None, left=None, right=None, value=None, none_value=None):
        self.name = name
        self.reference = reference
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.none_value = none_value # If this is not None, then this value is returned by predict if a feature value is None

    def predict(self, features):
        if self.value is not None:  # Leaf node
            return self.value, f"{self.value}"
        feature_value = features[self.name]
        if feature_value is None:
            if self.none_value is not None:
                return self.none_value, f"{self.name} None -> {self.value}"
            else:
                pred, path = self.left.predict(features) 
                return pred, f"{self.name} None -> " + path
        elif feature_value <= self.threshold:
            pred, path = self.left.predict(features)
            return pred, f"{self.name} {feature_value} <= {self.threshold} -> " + path
        else:
            pred, path = self.right.predict(features)
            return pred, f"{self.name} {feature_value} > {self.threshold} -> " + path            
        

def compute_feature(feature, arr:np.ndarray, reference_arr:np.ndarray = None, spacing:list = [1, 1, 1]) -> list:
    func_kwargs = {'roi_arr': arr, 'roi_arr2': reference_arr, 'spacing':spacing}
    feature_func = getattr(feature_functions, feature)
    new_kwargs = check_arguments(func_kwargs, list(inspect.getfullargspec(feature_func))[0])
    feature_val = feature_func(**new_kwargs)

    return feature_val
    
    
def check_arguments(arg_dict, key_list):
    # Where should I put this function, utils.py or something like that?
    keys = arg_dict.keys()

    new_dict = dict(arg_dict)
    for key in keys:
        if key not in key_list:
            del new_dict[key]
    return new_dict    
    
    
def check_overlap(mask1, mask2, iou_threshold):

    mask_intersection = np.logical_and(mask1, mask2) 
    mask_union = np.logical_or(mask1, mask2)  
    iou = np.sum(mask_intersection) / np.sum(mask_union)

    return 1 if iou >= iou_threshold else 0

def predict(tree, features):
    return tree.predict(features)


def normalize_tree_dict(tree_dict):
    """
    Normalizes various tree dict formats to a plotting-friendly structure.
    """
    if tree_dict is None:
        return None
    if isinstance(tree_dict, list):  # Leaf node
        return {"value": tree_dict}
    if "value" in tree_dict:  # Already normalized leaf
        return tree_dict

    reference = tree_dict.get("reference", tree_dict.get("_reference"))
    node = {
        "name": tree_dict.get("name"),
        "threshold": tree_dict.get("threshold"),
        "left": normalize_tree_dict(tree_dict["left"]),
        "right": normalize_tree_dict(tree_dict["right"]),
    }
    if reference is not None:
        node["reference"] = reference
    if "none_value" in tree_dict:
        node["none_value"] = tree_dict["none_value"]
    return node


def plot_tree(tree, x=0.5, y=1.0, x_offset=0.2, y_offset=0.1, parent_coords=None, edge_label=None):
    """
    Recursively plot a decision tree using matplotlib.
    """
    if "value" in tree:  # Leaf node
        plt.text(
            x,
            y,
            f"Leaf\n{tree['value']}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="lightgrey"),
        )
        if parent_coords:
            plt.gca().add_patch(FancyArrowPatch(parent_coords, (x, y), arrowstyle="-|>", color="black"))
            if edge_label:
                plt.text((parent_coords[0] + x) / 2, (parent_coords[1] + y) / 2, edge_label, fontsize=8)
    else:  # Internal node
        ref = f" [{tree['reference']}]" if tree.get("reference") else ""
        plt.text(
            x,
            y,
            f"{tree['name']}{ref} <= {tree['threshold']}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue"),
        )
        if parent_coords:
            plt.gca().add_patch(FancyArrowPatch(parent_coords, (x, y), arrowstyle="-|>", color="black"))
            if edge_label:
                plt.text((parent_coords[0] + x) / 2, (parent_coords[1] + y) / 2, edge_label, fontsize=8)

        # Plot children
        left_x = x - x_offset
        right_x = x + x_offset
        child_y = y - y_offset
        plot_tree(tree["left"], left_x, child_y, x_offset * 0.6, y_offset, (x, y), "True")
        plot_tree(tree["right"], right_x, child_y, x_offset * 0.6, y_offset, (x, y), "False")

def execute(
    candidate_masks: SMImage,
    relative_to_mask: SMImage,
    pydt,
    feature_name_list: list,
    learn: bool = False,
    ref_iou_threshold: float = 0.0
) -> Tuple[list, int]:
    """
    Performs decision tree analysis on candidate masks and returns a dictionary of DT output for each candidate.

    Args:
        candidate_masks (SMImage): Binary mask or multiple masks as from conn_comp_tool.
        relative_to_mask (SMImage, optional): Binary mask relative to which features are computed if applicable (e.g., for spatial relationships).
        
    Returns:
        dict: The decision tree output for each candidate.
            The dictionary d, has an item d['candidates'] that is a list of dictionaries (one dict for each candidate).
            Each candidate dictionary cand_dict, contains items: 
                cand_dict['name'] = 'cand_X' (where X is the mask number of the candidate)
                cand_dict['confidence'] = [0.0 to 1.0]
        
    config.json:   
    "decision_tree": {
        "code": "decision_tree_tool.py",
        "context": "./decision_tree_tool/",
        "parameters": {
            "candidate_masks": "from conn_comp",
            "relative_to_mask": "from trachea",
            "dt_json_path": "right_lung_dt.json"
        }
    },   
    """    
            
    if candidate_masks is None:
        return None, 0
        
    candidates_array = candidate_masks.pixel_array
    cand_max = int(np.max(candidates_array)) # the type conversion is done in case a mask (of type float) is passed rather than a label_mask
    #self.print_log(f"{cand_max} candidates received", sample_id)
    
    relative_to_array = None
    if relative_to_mask is not None:
        relative_to_array = relative_to_mask.pixel_array

    reasoning_output = None
    if cand_max>0: 
        reasoning_output = []     
        spacing = candidate_masks.metadata['spacing']
        for segid in range(1, cand_max + 1):
            cand = np.zeros(candidates_array.shape) # Initialize candidate array
            cand[candidates_array == segid] = 1 # Grab the candidate with the segid label

            cand_dict = {}
            cand_name = 'cand_{}'.format(segid)
            cand_dict['name'] = cand_name # Only need the candidate ID, no need to store all the arrays
            features = {}
            actual_val_list = []

            for feature_tuple in feature_name_list:
                feature = feature_tuple[0]
                actual_val = compute_feature(feature, arr = cand, 
                                                        reference_arr = relative_to_array, 
                                                        spacing = spacing)
                if isinstance(actual_val, bool):
                    actual_val=1 if actual_val else 0
                actual_val_list.append(actual_val)
                features[feature] = actual_val
            cand_dict['features'] = features
            
            prediction, pred_path = predict(pydt, features)
            cand_dict['confidence'] = prediction[1]
            cand_dict['prediction_path'] = pred_path                  

            if learn:
                cand_dict['ref_output'] = check_overlap(cand, candidate_masks.label_array, ref_iou_threshold) 
                            
            reasoning_output.append(cand_dict)
            
        # if not len(reasoning_output): # if no accepted candidates
        #     self.print_log("No candidates accepted based on provided rules.", sample_id)
            
    return reasoning_output, cand_max
