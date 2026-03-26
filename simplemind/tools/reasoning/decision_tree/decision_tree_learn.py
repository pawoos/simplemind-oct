"""
Tool Name: decision_tree_learn
=================================

Description:
    Trains a decision tree for classifying mask candidates. 
    To learn more about decision trees, see: https://gitlab.com/hoffman-lab/sm-incubator/-/blob/mbrown/sm_pipeline/docs/decision_trees.md

Parameters:            
    -pydt_dict (dict): Dictionary defining the decision tree.
        Can also be a path to a json file.
    - candidate_masks (SMImage): Training candidate masks that include the candidate masks as pixel_data and the reference mask as label_data.
        Mask values for each candidate increment from 1 (0 is background).
        These are typically input from the conn_comp tool.
        They are passed through by the `execute` method and are aggregated into the `results` arg of the `aggregate` method, where training is performed.
    - relative_to_mask (SMImage, optional): Binary mask relative to which features are computed if applicable (e.g., for spatial relationships).
    - ref_iou_threshold (float, optional): The interestion over union threshold between the candidate and the reference for the candidate to be considered a match. Default = 0.7.
    - max_depth (int, optional): Maximum depth of the trained decision tree. Default = None.
    - output_dir (str): Where training outputs will be written. Default = "./decision_tree".
    - visualize_png (bool_: Write a file to visualize the trained tree. Default = False,
    - learn_output_name (str): The file name for the trained DT in json format. Default = "dt_train".        
            
Output:
    - None (DT json and other output files are not directly accessible to other tools)
            
Example JSON Plan:
    "candidate_selection-decision_tree": {
        "code": "decision_tree.py",
        "code_learn": "decision_tree_learn.py",
        "context": "./tools/reasoning/decision_tree/",
        "candidate_masks": "from candidate_selection-conn_comp",
        "relative_to_mask": "from mask_processing-morph_close",
        "pydt_dict": "plan/cxr/right_lung_dt.json",
        "max_depth": 1,
        "output_dir": "../output",
        "visualize_png": true
    }
    
Notes:
    - Typically used within a candidate_selection chunk, along with conn_comp and cand_select tools.
    - torch_segmentation_learn writes weights and log files, but no output to the Blackboard, i.e., no output that can be used as input to other tools
"""

import asyncio
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sm_sample_aggregator import SMSampleAggregator
from sm_sample_id import SMSampleID
from sm_image import SMImage
import dt_helper


class DecisionTreeLearn(SMSampleAggregator):
        
    async def setup(
        self,
        *,
        pydt_dict: dict,
    ) -> None:
        """
        Initializes the decision tree.
        """   
        self.pydt, self.feature_name_list = dt_helper.setup(pydt_dict)
        #self.print_error(f"feature_name_list: {self.feature_name_list}")
            
        
    async def execute(
        self,
        *,
        candidate_masks: SMImage,
        relative_to_mask: SMImage = None,
        ref_iou_threshold: float = 0.7,
        sample_id: SMSampleID
    ) -> list:

        reasoning_output, cand_max = dt_helper.execute(candidate_masks, relative_to_mask, self.pydt, self.feature_name_list, True, ref_iou_threshold)
        
        if reasoning_output is None:
            reasoning_output = []
                
        self.print_log(f"{cand_max} candidates received", sample_id)
                
        return reasoning_output
    

    async def aggregate(
        self,
        *,
        dataset_id: str,
        results: list,
        total: int,
        output_dir: str = "./decision_tree",
        max_depth: int = None,
        visualize_png: bool = False,
        learn_output_name: str = "dt_train",
        #msg_tags: list[str],
    ) -> None:
        # Candidate features have been computed for all cases in the dataset
        # Form training samples
        training_samples = []
        for item in results:
             training_samples.extend(item)
        self.print_log(f"training_samples: {training_samples}")
            
        #X_train = training_data[feature_list]
        #y_train = training_data['overlap_flag']
        feature_names = self.feature_name_list

        #samples = my_agent_results['candidates']
        # Extract the features and convert them to a DataFrame
        X_train = pd.DataFrame([sample['features'] for sample in training_samples])
        #await self.log(Log.INFO, f"LEARNING: X_train = {X_train}")
        self.print_log(f"X_train = {X_train}")

        # Extract the labels (if needed for `y`)
        y_train = [sample['ref_output'] for sample in training_samples]
        #await self.log(Log.INFO, f"LEARNING: y_train = {y_train}")
        self.print_log(f"y_train = {y_train}")

        clf = DecisionTreeClassifier()
        if max_depth is not None:
            clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, y_train)
        #tree_rules = export_text(clf, feature_names=feature_names)
        #await self.log(Log.INFO, f"LEARNING: tree_rules = {tree_rules}")

        #learn_output_name = agent_parameters['learn_output_name']
        tree_dict = self.sklearntree_to_dict(clf, feature_names)
        op_dir = self.resolve_output_dir(output_dir, dataset_id)
        os.makedirs(op_dir, exist_ok=True)
        base_tool_name = self.name().replace(f"-{self.plan_id}", "")
        json_filename = os.path.join(op_dir, f"{base_tool_name}_{learn_output_name}.json")
        #await self.log(Log.INFO, f"LEARNING: tree_dict = {tree_dict}")
        #await self.log(Log.INFO, f"LEARNING: tree_dict['threshold'] = {tree_dict['threshold']}")
        tree_dict = self.convert_numpy_to_native(tree_dict)
        with open(json_filename, "w") as json_file:
            json.dump(tree_dict, json_file, sort_keys=False, indent=2)
            #yaml.dump(tree_dict, yaml_file, default_flow_style=False, sort_keys=False, default_style=None)

        if visualize_png:
            dt_train_png_path = os.path.join(op_dir, f"{base_tool_name}_{learn_output_name}.png")
            plt.figure(figsize=(10, 6))
            dt_helper.plot_tree(dt_helper.normalize_tree_dict(tree_dict))
            plt.axis("off")
            plt.savefig(dt_train_png_path, dpi=300, bbox_inches="tight")
            plt.close()

    def sklearntree_to_dict(self, tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!" for i in tree_.feature
        ]

        def recurse(node):
            if tree_.feature[node] != -2:
                if feature_name[node][1] is not None:
                    return {
                        "name": feature_name[node][0],
                        "reference": feature_name[node][1],
                        "threshold": tree_.threshold[node],
                        "left": recurse(tree_.children_left[node]),
                        "right": recurse(tree_.children_right[node]),
                    }
                else:
                    return {
                        "name": feature_name[node][0],
                        "threshold": tree_.threshold[node],
                        "left": recurse(tree_.children_left[node]),
                        "right": recurse(tree_.children_right[node]),
                    }                    
            else:
                #return {"value": tree_.value[node]}
                #return {"value": tree_.value[node].tolist()}
                return {"value": str(tree_.value[node].tolist())[1:-1]}

        return recurse(0)

    def convert_numpy_to_native(self, data):
        """
        Recursively convert NumPy types in a dictionary to native Python types.
        """
        if isinstance(data, dict):
            return {key: self.convert_numpy_to_native(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_numpy_to_native(item) for item in data]
        elif isinstance(data, np.generic):
            return data.item()  # Convert NumPy scalar to native Python type
        else:
            return data
        
if __name__ == "__main__":   
    tool = DecisionTreeLearn()
    asyncio.run(tool.main())
