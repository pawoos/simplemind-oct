"""
Tool Name: cand_select
=================================

Description:
    Selects candidate masks either by thresholding confidences and/or selecting the largest connected candidate.
    All satisfying candidates are combined into a single binary mask.
    If no candidates are accepted, then the output will be None, unless output_empty_mask is True.

Parameters:            
    - candidate_masks (SMImage): Mask values for each candidate increment from 1 (0 is background).
        These are typically input from the conn_comp tool.
    - candidate_confidences (dict, optional): The decision tree output for each candidate.
        The dictionary d, has an item d['candidates'] that is a list of dictionaries (one dict for each candidate).
        Each candidate dictionary, cand_dict, contains items: 
            cand_dict['name'] = 'cand_X' (where X is the mask number of the candidate)
            cand_dict['confidence'] = [0.0 to 1.0]
        If this argument is None, then it is expected that largest_only will be True, otherwise None will be returned. 
    - threshold (float, optional): Candidates are selected if their confidence is >= this threshold. Default = 0.5.
    - largest_only (bool, optional): If this is True the select only the largest candidate (most voxels). Default = false.
        If candidate_confidences is None then select the largest.
        Otherwise, select the largest of the candidates with confidences above the threshold.
    - output_empty_mask (bool, optional): If true and no candidates are selected then an empty mask (all zeros). Default = false.
        If false (default) amd no candidates are selected then None is output.

Output:
    - SMImage: Selected candidates are output as a single binary mask.
            
Example JSON Plan:
    "candidate_selection-cand_select": {
        "code": "cand_select.py",
        "context": "./tools/reasoning/cand_select/",
        "candidate_masks": "from candidate_selection-conn_comp",
        "candidate_confidences": "from candidate_selection-decision_tree",
        "threshold": 0.5,
        "largest_only": true
    }
    
Notes:
    - Typically used within a candidate_selection chunk, along with conn_comp and decision_tree tools.
"""

import asyncio
import numpy as np
import pandas as pd

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

class CandSelect(SMSampleProcessor):

    async def execute(
        self,
        *,
        candidate_masks: SMImage,
        #candidate_confidences: dict = None,
        candidate_confidences: list = None,
        threshold: float = 0.5,
        largest_only: bool = False,
        output_empty_mask: bool = False,
        sample_id: SMSampleID
        #msg_tags: list[str]
    ) -> SMImage: 

        if candidate_masks is None:
            return None
        
        self.print_log(f"candidate_confidences: {candidate_confidences}", sample_id)
        selected_candidate_array = self.select_candidate(candidate_masks.pixel_array, cand_confs=candidate_confidences, largest_only=largest_only, 
                                                        thres=threshold,accept_blank_image=output_empty_mask)

        if selected_candidate_array is not None:
            return SMImage(candidate_masks.metadata, selected_candidate_array, candidate_masks.label_array)
        else:
            return None


    def select_candidate(self, cand_arr: np.ndarray, cand_confs: list, largest_only: bool = False, thres: float = 0.5, accept_blank_image: bool = False) -> np.ndarray:

        N = int(np.max(cand_arr))
        accepted_candidates = [] # Hopefully accepted candidates won't be too many (ŏ_ŏ)
        if cand_confs is None:
            return None
        
        if len(cand_confs) > 0:
            confidence_scores = list(pd.DataFrame(cand_confs)['confidence']) # Feels like a stupid use of pandas tbh

            #TODO better method, add accepted candidates to the same array then use cc3d to select largest if combine = False
            for segid, confidence in zip(range(1, N + 1), confidence_scores):
                
                if confidence >= thres:
                    accepted_candidate = (cand_arr == segid).astype(np.uint8)
                    accepted_candidates.append(accepted_candidate)

        if len(accepted_candidates)==0: # If no candidate is accepted, return None
            if accept_blank_image:
                return np.zeros(cand_arr.shape, dtype=np.uint8)
            else:
                return None
        
        if not largest_only:
            best_candidate = sum(accepted_candidates) # Place all accepted candidates in the same array

        else: # If combine is false, grab the largest candidate
            voxel_num = 0
            for candidate in accepted_candidates:
                voxels_in_candidate = np.count_nonzero(candidate)
                if  voxels_in_candidate > voxel_num:
                    best_candidate = candidate  # Grab largest candidate

                    voxel_num = voxels_in_candidate

        return best_candidate


if __name__ == "__main__":   
    tool = CandSelect()
    asyncio.run(tool.main())
