"""
Tool Name: decision_tree
=================================

Description:
    Performs decision tree analysis on candidate masks and returns a dictionary of DT output for each candidate.        
    To learn more about decision trees, see: https://gitlab.com/hoffman-lab/sm-incubator/-/blob/mbrown/sm_pipeline/docs/decision_trees.md

Parameters:            
    -pydt_dict (dict): Dictionary defining the decision tree.
        Can also be a path to a json file.
    - candidate_masks (SMImage): Mask values for each candidate increment from 1 (0 is background).
        These are typically input from the conn_comp tool.
    - relative_to_mask (SMImage, optional): Binary mask relative to which features are computed if applicable (e.g., for spatial relationships).
    - output_dir (str, optional): Base directory for optional visualizations (default is the tool output dir).
    - visualize_png (bool, optional): Write a PNG visualization of the decision tree once per dataset. Default = False.
    - visualize_png_dir (str, optional): Subdirectory (relative to base output dir) to write the visualization. Default = "./decision_tree".
    - visualize_png_name (str, optional): Base filename (without extension) for the visualization. Default = "dt".
    - log_reasoning_output (bool, optional): Write reasoning output to a log file per sample. Default = False.
    - log_reasoning_dir (str, optional): Subdirectory (relative to base output dir) for reasoning logs. Default = "./samples".
    - log_reasoning_filename (str, optional): Base filename for reasoning logs. Default = None (uses tool name without plan id).
            
Output:
    - dict: The decision tree output for each candidate.
        The dictionary d, has an item d['candidates'] that is a list of dictionaries (one dict for each candidate).
        Each candidate dictionary cand_dict, contains items: 
            cand_dict['name'] = 'cand_X' (where X is the mask number of the candidate)
            cand_dict['confidence'] = [0.0 to 1.0]
            
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
    - An initial version of the decision tree is usually created by hand and then refined by training.
    - For details on the decision tree, see the Decision Tree tutorial exercise.  
"""

import asyncio
import os
import json

from sm_sample_processor import SMSampleProcessor
from sm_sample_id import SMSampleID
from sm_image import SMImage
import dt_helper
import matplotlib.pyplot as plt

class DecisionTree(SMSampleProcessor):
        
    async def setup(
        self,
        *,
        pydt_dict: dict,
        output_dir: str | None = "./decision_tree",
        visualize_png: bool = False,
        visualize_png_dir: str = "./decision_tree",
        visualize_png_name: str = "dt",
        log_reasoning_output: bool = False,
        log_reasoning_dir: str = "./samples",
        log_reasoning_filename: str | None = None,
    ) -> None:
        """
        Initializes the decision tree.
        """  
            
        try:
            #self.pydt, self.feature_name_list = dt_helper.setup(dt_json_path)
            self.pydt, self.feature_name_list = dt_helper.setup(pydt_dict)
        except Exception as exc:
            self.print_error(f"Invalid decision tree definition (pydt_dict): {exc}. Check that each node has required keys like 'left' and 'right'.")
            raise
        self.output_dir = output_dir
        self.visualize_png = visualize_png
        self.visualize_png_dir = visualize_png_dir
        self.visualize_png_name = visualize_png_name
        self._visualized_datasets = set()
        self.tree_dict = dt_helper.normalize_tree_dict(pydt_dict) if isinstance(pydt_dict, (dict, list)) else None
        self.log_reasoning_output = log_reasoning_output
        self.log_reasoning_dir = log_reasoning_dir
        self.log_reasoning_filename = log_reasoning_filename
            
       
    async def execute(
        self,
        *,
        candidate_masks: SMImage,
        relative_to_mask: SMImage = None,
        sample_id: SMSampleID
    ) -> list:

        reasoning_output, cand_max = dt_helper.execute(candidate_masks, relative_to_mask, self.pydt, self.feature_name_list)
                
        self.print_log(f"{cand_max} candidates received", sample_id)
        self.print_log(f"reasoning_output: {reasoning_output}", sample_id)
        if reasoning_output is not None: # if no accepted candidates
            highest_conf_cand = max(reasoning_output, key=lambda x: x["confidence"])
            self.print_log(f"Highest confidence candidate = {highest_conf_cand}", sample_id)
        self.maybe_log_reasoning_output(reasoning_output, sample_id)
        self.maybe_visualize_tree(sample_id)
                
        return reasoning_output

    def maybe_visualize_tree(self, sample_id: SMSampleID) -> None:
        if not self.visualize_png or self.tree_dict is None:
            return
        dataset_id = sample_id.dataset if sample_id else None
        if dataset_id in self._visualized_datasets:
            return

        out_dir = self.resolve_output_dir(self.visualize_png_dir, dataset_id)
        os.makedirs(out_dir, exist_ok=True)
        base_tool_name = self.name().replace(f"-{self.plan_id}", "")
        filename = f"{base_tool_name}.png" if self.visualize_png_name == "dt" else f"{self.visualize_png_name}.png"
        png_path = os.path.join(out_dir, filename)

        plt.figure(figsize=(10, 6))
        dt_helper.plot_tree(self.tree_dict)
        plt.axis("off")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        self._visualized_datasets.add(dataset_id)

    def maybe_log_reasoning_output(self, reasoning_output, sample_id: SMSampleID) -> None:
        if not self.log_reasoning_output or sample_id is None:
            return
        out_dir = self.resolve_output_dir(self.log_reasoning_dir, sample_id.dataset)
        sample_dir = self.sample_output_path(out_dir, sample_id)
        base_name = (
            self.name().replace(f"-{self.plan_id}", "")
            if self.log_reasoning_filename is None
            else self.log_reasoning_filename
        )
        log_path = os.path.join(sample_dir, f"{base_name}.log")
        try:
            pretty_reasoning = json.dumps(reasoning_output, indent=2)
            with open(log_path, "w") as f:
                f.write(f"dataset={sample_id.dataset}, sample={sample_id.sample}, total={sample_id.total}\n")
                f.write("reasoning_output:\n")
                f.write(pretty_reasoning)
                f.write("\n")
        except Exception as exc:
            self.print_error(f"Failed to write reasoning log to {log_path}: {exc}", sample_id, warning=True)


if __name__ == "__main__":   
    tool = DecisionTree()
    asyncio.run(tool.main())
