# Tool Documentation

### biomech_video_mask_reader
- [biomech_video_mask_reader.py](#biomech_video_mask_readerpy)

### image_processing
- [clahe.py](#clahepy)
- [minmax_norm.py](#minmax_normpy)
- [read_sm_image.py](#read_sm_imagepy)
- [resize.py](#resizepy)
- [save_png.py](#save_pngpy)
- [threshold.py](#thresholdpy)

### mask_processing
- [bounding_box.py](#bounding_boxpy)
- [conn_comp.py](#conn_comppy)
- [image_mask copy.py](#image_maskcopypy)
- [image_mask.py](#image_maskpy)
- [mask_features.py](#mask_featurespy)
- [mask_logic.py](#mask_logicpy)
- [morphology.py](#morphologypy)
- [spatial_offset.py](#spatial_offsetpy)

### neural_net
- [medsam2.py](#medsam2py)
- [torch_segmentation.py](#torch_segmentationpy)
- [torch_segmentation_learn.py](#torch_segmentation_learnpy)

### reasoning
- [cand_select.py](#cand_selectpy)
- [decision_tree.py](#decision_treepy)
- [decision_tree_learn.py](#decision_tree_learnpy)

***
## biomech_video_mask_reader

### biomech_video_mask_reader.py

```
Tool Name: biomech_video_mask_reader
=================================

Description:
    Reads a byte stream of csv data containing pose coordinates into SMImage masks.
    
    The csv format has 3 rows of header and the remainder are data.
        Row 2 has the object name, Row 3 indicates whether the column data is “x”, “y”, or “likelihood” for each object point.
        The first column is the z-coordinate for all points.       

Parameters:            
    - csv_bytes (bytes): Input csv stream.
    - x_dim (int): X dimension of the output mask.
    - y_dim (int): Y dimension of the output mask.
    - likelihood_threshold (float): Minimum likelihood to include a point.
        For each point with likelihood greater than or equal to the threshold, set a point in the mask to 1.
    - object_names (list[str]): List of object names to return.

Output:
    - Blackboard messages, one message for each object SMImage.
            
Example JSON Plan:
    "dataset_upload": {
        "code": "dataset_upload.py",
        "chunk": "load_image",
        "csv_path": "from arg dataset_csv",
        "post_tags": ["biomech_video_mask"]
    },

Notes:
    - The mask data is stored in the image data (pixel_array) of SMImage.
    - The coordinate values are rounded to integers.
    - This is not a typical tool, it implements a run method and posts messages to the Blackboard.
    - Most tools implement setup, execute, and/or aggregate methods and return values (they responsible for posting to the Blackboard).
    - This tool posts outputs with a "result" tag to be compatible with other SM pipeline tools.
```

## image_processing

### clahe.py

```
Tool Name: clahe
=================================

Description:
    Applies contrast limited adaptive histogram equalization (CLAHE) to an SMImage..

Parameters:            
    - input_image (SMImage): Input image to be processed.
    - nbins (int, optional): Number of gray bins for histogram (“data range”).
    - clip_limit (float, optional): Clipping limit, normalized between 0 and 1 (higher values give more contrast).

Output:
    - SMImage: The processed image.
            
Example JSON Plan:
    "clahe": {
        "code": "clahe.py",
        "context": "./tools/clahe/",
        "input_image": "from read_sm_image",
        "nbins": 256,
        "clip_limit": 0.03
    }

Notes:
    - The output image will be converted to the same type as the input image after CLAHE.
    - Internally uses `skimage.exposure.equalize_adapthist`.
    - For parameter details, see: https://scikit-image.org/docs/0.25.x/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
```

### minmax_norm.py

```
Tool Name: minmax_norm
=================================

Description:
    Apply min-max normalization to an SMImage. The normalized image is min value of 0.0 and max value of 1.0.

Parameters:            
    - input_image (SMImage): Input image to be normalized.

Output:
    - SMImage: The normalized image.
            
Example JSON Plan:
    "minmax_norm": {
        "code": "minmax_norm.py",
        "input_image": "from read_sm_image",
    }

Notes:
    - If all values are uniform in the input image, then the result is all 0.
```

### read_sm_image.py

```
Tool Name: read_sm_image
=================================

Description:
    Reads a byte stream of array data into an SMImage.

Parameters:            
    - image_bytes (bytes): Input image array.
    - label_mask_bytes (bytes, optional): Label array.
        Expected to be int values representing each mask of the dimensions as the image_bytes.

Output:
    - SMImage.
            
Example JSON Plan:
    "read_sm_image": {
        "code": "read_sm_image.py",
        "image_bytes": "from dataset_upload image file",
        "label_mask_bytes": "from dataset_upload mask file",
        "final_output": true
    }

Notes:
    - Typically provides the final_output for an input_image_plan.
```

### resize.py

```
Tool Name: resize
=================================

Description:
    Resize an SMImage to either a specified `target_shape` or to match the shape of a given `target_image`.

Parameters:            
    - input_image (SMImage): Input image to be resized.
    - target_shape (tuple, optional): Desired shape (z, y, x) in NumPy order. Must match input dimensionality.
        If the shape does not match, an empty message will be posted.
    - target_image (SMImage, optional): If provided, resizes to match this image shape.
    - order (int, optional): Logging verbosity. Spline interpolation order (0 to 5). Default is 3..
    - preserve_range (bool, optional): Preserve original image value range. Default is True.
    - anti_aliasing (bool, optional): Apply anti-aliasing during rescaling. Default is True.

Output:
    - SMImage: The resized image.
            
Example JSON Plan:
    "image_preprocessing-resize": {
        "code": "resize.py",
        "input_image": "from input_image",
        "target_shape": [1, 512, 512],
        "order": 3,
        "preserve_range": true,
        "anti_aliasing": true
    }

Notes:
    - If both `target_shape` and `target_image` are provided, `target_image` takes priority.
    - The output image will have the same type as the input image.
    - Internally uses `skimage.transform.resize`.
    - For parameter details, see: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
```

### save_png.py

```
Tool Name: save_png
=================================

Description:
    Generate a png file from an SMImage.
        For 3D, selected 2D images are included.
        Works for images and masks as the input_image.pixel_array.
        Can display the input_image.label_array as an overlay.

Parameters:            
    - input_image (SMImage): Input image to be saved as png.
    - input_mask (SMImage, optional): Displayed as an image overlay. Default = None.
        input_mask.pixel_array is the mask.
        Supersedes input_image.label_array as the overlay.
    - output_dir (str, optional): Path to which png will be saved. Default is working-<id>/output/samples/0/.
        Current directory (.) is the pipeline working directory.
    - filename (str, optional): If provided, then this will be the name of the png file.
        Default is object_chunk_tool.png.
    - show_label (bool, optional): Display the input_image label as an overlay (if it exists). Default = True.
    - flatten_axis (int, optional): Flatten an axis if 3D. Default = None.
    - mask_slice_axis (int, optional): If an input_mask is provided, show only the slice of the mask with the most pixels. 
        0=z-axis, 1=y, 2=x.
        Default = None.
        This takes priority over flatten if both are provided.
    - invert_mask (bool, optional): If true, invert masks. Default = false.
        Invert mask images and label overlays from white to black foreground.
    - mask_color (str, optional): Default is red.
        Options: 'pink', 'green', 'torquise', 'cyan', 'blue', 'brightGreen', 'orange', 
        'red', 'yellow', 'purple', 'magenta', 'parrotGreen', 'b_green', 'yellow_green', 'light_blue'
    - mask_alpha (float, optional): 0.0 is most transparent, 1.0 is solid. Default = 0.8.
    - title (str, optional): Appears at the top of the plot. Default = None.
    - mask_none (bool, optional): Save an image even if the mask/label is None. Default = True.
        Otherwise, no image is saved if the mask/label is None.

Output:
    - None (that can be used by other tools).
            
Example JSON Plan:
    "save_png": {
        "code": "save_png.py",
        "input_image": "from any SMImage",
        "output_dir": "../output"
    }
  
Notes:
    - save_png writes a png file, but no output to the Blackboard, i.e., no output that can be used as input to other tools.
```

### threshold.py

```
Tool Name: threshold
=================================

Description:
    Performs thresholding on an SMImage and returns a binary mask as an SMImage.

Parameters:            
    - input_image (SMImage): Input image to be thresholded.
    - lower_threshold (float, optional): Lower threshold bound.
    - upper_threshold (float, optional): Upper threshold bound.

Output:
    - SMImage: The thresholded mask.
            
Example JSON Plan:
    "threshold": {
        "code": "threshold_tool.py",
        "context": "./threshold_tool/",
        "input_image": "from clahe",
        "upper_threshold": 0.4
    }

Notes:
    - Thresholds use >= and <=.
    - lower_threshold and upper_threshold are both optional, but at least one must be provided.
```

## mask_processing

### bounding_box.py

```
Tool Name: bounding_box
=================================

Description:
    Computes a bounding box mask around nonzero regions of an SMImage mask.
    Supports voxel, mm, and length-fraction offset units, as well as slice-wise bounding boxes.

Parameters:            
    - input_image (SMImage): Input binary mask image.
    - z_upper_offset (float, optional): Offset added to the upper z coordinate.
    - z_lower_offset (float, optional): Offset added to the lower z coordinate.
    - y_upper_offset (float, optional): Offset added to the upper y coordinate.
    - y_lower_offset (float, optional): Offset added to the lower y coordinate.
    - x_upper_offset (float, optional): Offset added to the upper x coordinate.
    - x_lower_offset (float, optional): Offset added to the lower x coordinate.
    - offset_unit (str, optional): One of "mm", "voxels" (default), or "length_fraction".
    - slice_wise_bounding_box (bool, optional): If True, compute slice-wise 2D bounding boxes.
    - axis (str, optional): Axis for slice-wise bounding boxes ("z", "y", or "x"). Default = "z".

Output:
    - SMImage: The bounding box mask image.

Example JSON Plan:
    "bounding_box": {
        "code": "bounding_box.py",
        "input_image": "from mask_processing-morph_close",
        "z_upper_offset": 5,
        "offset_unit": "mm"
    }
    
Notes:
    - If the input mask is empty then the output mask will also be empty.
```

### conn_comp.py

```
Tool Name: conn_comp
=================================

Description:
    Performs connected component analysis on an SMImage mask and returns a mask as an SMImage.
    Each connected component is assigned an integer value, starting from 1 (0 is background).

Parameters:            
    - input_image (SMImage): Input image to be processed.
    - connectivity (int, optional): Only 4,8 (2D) and 26, 18, and 6 (3D) are allowed. Default = 6.
    - voxel_count_threshold (int, optional): Only include components with this number of voxels or more.
    - binary_mask (bool, optional): If true then all components (meeting voxel_count_threshold) are returned as a single binary mask.
        Default = False.

Output:
    - SMImage: The connected component mask image.
            
Example JSON Plan:
    "candidate_selection-conn_comp": {
        "code": "conn_comp.py",
        "input_image": "from mask_processing-morph_close",
        "voxel_count_threshold": 100
    }

Notes:
    - Uses the cc3d package: https://pypi.org/project/connected-components-3d/
```

### image_mask copy.py

```
Tool Name: image_mask
=================================

Description:
    Creates a mask image by filling a rectangular region defined by proportional
    bounds along the x, y, and z axes. Bounds are specified as proportions of 
    the respective dimension size (values in [0.0, 1.0]).

Parameters:            
    - input_image (SMImage): Input image (mask or image) used for dimensions.
    - x_upper_prop (float, optional): Upper x-axis proportion bound. Default = 0.
    - x_lower_prop (float, optional): Lower x-axis proportion bound. Default = 0.
    - y_upper_prop (float, optional): Upper y-axis proportion bound. Default = 0.
    - y_lower_prop (float, optional): Lower y-axis proportion bound. Default = 0.
    - z_upper_prop (float, optional): Upper z-axis proportion bound. Default = 0.
    - z_lower_prop (float, optional): Lower z-axis proportion bound. Default = 0.

Output:
    - SMImage: The generated rectangular mask image.

Example JSON Plan:
    "mask_generation-image_mask": {
        "code": "image_mask.py",
        "input_image": "from preprocessing-normalize",
        "x_upper_prop": 0.8,
        "x_lower_prop": 0.2,
        "y_upper_prop": 0.9,
        "y_lower_prop": 0.1,
        "z_upper_prop": 1.0,
        "z_lower_prop": 0.0
    }
```

### image_mask.py

```
Tool Name: image_mask
=================================

Description:
    Creates a mask image by filling a rectangular region defined by proportional
    bounds along the x, y, and z axes. Bounds are specified as proportions of 
    the respective dimension size (values in [0.0, 1.0]).

Parameters:            
    - input_image (SMImage): Input image (mask or image) used for dimensions.
    - x_upper_prop (float, optional): Upper x-axis proportion bound. Default = 1.
    - x_lower_prop (float, optional): Lower x-axis proportion bound. Default = 0.
    - y_upper_prop (float, optional): Upper y-axis proportion bound. Default = 1.
    - y_lower_prop (float, optional): Lower y-axis proportion bound. Default = 0.
    - z_upper_prop (float, optional): Upper z-axis proportion bound. Default = 1.
    - z_lower_prop (float, optional): Lower z-axis proportion bound. Default = 0.

Output:
    - SMImage: The generated rectangular mask image.

Example JSON Plan:
    "mask_generation-image_mask": {
        "code": "image_mask.py",
        "input_image": "from preprocessing-normalize",
        "x_upper_prop": 0.8,
        "x_lower_prop": 0.2,
        "y_upper_prop": 0.9,
        "y_lower_prop": 0.1,
        "z_upper_prop": 1.0,
        "": 0.0
    }
    
Notes:
    - z_upper_prop: proportion [0.0 - 1.0] of the image z-axis dimension that provides the upper z-axis bound of the output mask 
        - if not provided then the z_upper_prop is 1.0 by default (to cover the entire image with the mask)
    - z_lower_prop: proportion [0.0 - 1.0] of the image z-axis dimension that provides the lower z-axis bound of the output mask; 
        - if not provided then the z_lower_prop is 0 by default (to cover the entire image with the mask)
```

### mask_features.py

```
Tool Name: mask_features
=================================

Description:
    Computes a dictonary of mask features.

Parameters:            
    - input_mask (SMImage): Input binary mask image.

Output:
    - dictionary of mask features: {
        centroid:
    }

Example JSON Plan:
    "bounding_box": {
        "code": "bounding_box.py",
        "input_image": "from mask_processing-morph_close",
        "z_upper_offset": 5,
        "offset_unit": "mm"
    }
    
Notes:
    -
```

### mask_logic.py

```
Tool Name: mask_logic
=================================

Description:
    Performs logical operations on one or two SMImage masks and returns the resulting mask as an SMImage.
    Supported logical operators: 'and', 'or', 'not', 'xor', 'sub', 'ifnot', 'ifor', 'incontact'.

Parameters:
    - input_1 (SMImage): First input mask (required).
    - input_2 (SMImage, optional): Second input mask, used depending on operator.
    - logical_operator (str): Logical operator to apply.
    - none_if_empty (bool, optional): If True and the result mask is empty, returns None. Default = False.

Output:
    - SMImage: The resulting mask after applying the logical operation.

Example JSON Plan:
    "mask_logic-example": {
        "code": "mask_logic.py",
        "input_1": "from some_previous_step",
        "input_2": "from another_step",
        "logical_operator": "and",
        "none_if_empty": True
    }
```

### morphology.py

```
Tool Name: morphology
=================================

Description:
    Performs morphological processing on binary masks (SMImage) in 2D or 3D.
    Masks are assumed to contain only 0 and 1 values.

Parameters:
    - input_image (SMImage): Input binary mask to be processed (2D or 3D).
    - morphological_task (str): One of "open", "close", "erode", "dilate".
    - kernel (str): Specifies the structuring element.
        * For 2D kernels: "rectangle width height", "ellipse width height"
        * For 3D kernels: "rectangle depth height width", "ellipse depth height width", "ball radius"
        * For a rectangle with range of [-x, x] use size parameter 2x+1
    - dimensionality (int, optional): 2 or 3 (default = 2).
        Determines whether to apply morphology in 2D or 3D.

Output:
    - SMImage: The processed binary mask (values 0 and 1).

Example JSON Plan:
    # 2D morphological closing with ellipse
    "mask_processing-morph_close_2d": {
        "code": "morphology.py",
        "context": "./tools/mask_processing/morphology/",
        "input_image": "from neural_net-torch_seg",
        "morphological_task": "close",
        "kernel": "ellipse 10 10",
        "dimensionality": 2
    }

    # 3D morphological closing with a ball kernel
    "mask_processing-morph_close_3d": {
        "code": "morphology.py",
        "context": "./tools/mask_processing/morphology/",
        "input_image": "from neural_net-torch_seg",
        "morphological_task": "close",
        "kernel": "ball 5",
        "dimensionality": 3
    }

Notes:
    - Input and output masks must contain only values 0 and 1.
    - Uses skimage.morphology (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html).
    - 2D supported kernels: "rectangle", "ellipse".
    - 3D supported kernels: "rectangle", "ellipse", "ball".
```

### spatial_offset.py

```
Tool Name: spatial_offset
=================================

Description:
    Creates a subregion mask around the centroid of the input mask with specified
    spatial offsets along the x, y, and z axes. Offsets can be specified in either
    voxel units or millimeters.

Parameters:            
    - input_image (SMImage): Input binary mask image.
    - x_offset_1 (float, optional): Lower x bound offset from centroid.
    - x_offset_2 (float, optional): Upper x bound offset from centroid.
    - y_offset_1 (float, optional): Lower y bound offset from centroid.
    - y_offset_2 (float, optional): Upper y bound offset from centroid.
    - z_offset_1 (float, optional): Lower z bound offset from centroid.
    - z_offset_2 (float, optional): Upper z bound offset from centroid.
    - offset_unit (str, optional): One of "mm" or "voxels" (default: "voxels").

Output:
    - SMImage: The cropped subregion mask.

Example JSON Plan:
    "spatial_offset": {
        "code": "spatial_offset.py",
        "input_image": "from bounding_box",
        "x_offset_1": -20,
        "x_offset_2": 20,
        "y_offset_1": -15,
        "y_offset_2": 15,
        "z_offset_1": -10,
        "z_offset_2": 10,
        "offset_unit": "mm"
    }
```

## neural_net

### medsam2.py

```
Tool Name: resize
=================================

Description:
    Resize an SMImage to either a specified `target_shape` or to match the shape of a given `target_image`.

Parameters:            
    - input_image (SMImage): Input image to be resized.
    - target_shape (tuple, optional): Desired shape (z, y, x) in NumPy order. Must match input dimensionality.
        If the shape does not match, an empty message will be posted.
    - target_image (SMImage, optional): If provided, resizes to match this image shape.
    - order (int, optional): Logging verbosity. Spline interpolation order (0 to 5). Default is 3..
    - preserve_range (bool, optional): Preserve original image value range. Default is True.
    - anti_aliasing (bool, optional): Apply anti-aliasing during rescaling. Default is True.

Output:
    - SMImage: The resized image.
            
Example JSON Plan:
    "image_preprocessing-resize": {
        "code": "resize.py",
        "input_image": "from input_image",
        "target_shape": [1, 512, 512],
        "order": 3,
        "preserve_range": true,
        "anti_aliasing": true
    }

Notes:
    - If both `target_shape` and `target_image` are provided, `target_image` takes priority.
    - The output image will have the same type as the input image.
    - Internally uses `skimage.transform.resize`.
    - For parameter details, see: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
```

### torch_segmentation.py

```
Tool Name: torch_segmentation
=================================

Description:
    Runs segmentation inference using a deep convolutional neural network (CNN).

Parameters:            
    - input_image (SMImage): Input image for the prediction.
    - model_backbone (str): Currently supported models are defined in models/unet_zoo.py.
    - weights_path (str): Path to the .pth file.
        Be careful not to commit the weights file to the git repo (it is too large).
    - prediction_threshold (float, optional): Applied to the predition map generated by the CNN to obtain a binary segmentation mask. Default = 0.5.
    - num_classes (int): number of class labels to be included in the segmentation prediction (typically 1).
    - model_specific_output_class (int, optional): Class to be returned if multiple classes are included in the segmentation model.
    - map_output_dir (str, optional): Provide this if the prediction map is to be written as a png.
        The filename will be based on the tool name and end with `_map.png`.
    - gpu_num: GPU to be used for computation. Default = 0, i.e., use CPU.

Output:
    - SMImage: Binary segmentation mask.
            
Example JSON Plan:
  "neural_net-torch_seg": {
    "code": "torch_segmentation.py",
    "code_learn": "torch_segmentation_learn.py",
    "input_image": "from image_preprocessing-clahe",
    "model_backbone": "resnet50",
    "weights_path": "/home/matt/weights/right_lung_weights.pth",
    "num_classes": 1,
    "map_output_dir": "from arg output_dir",
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 100
  }
  
Notes:
    - Uses the pytorch package: https://docs.pytorch.org/docs/stable/
```

### torch_segmentation_learn.py

```
Tool Name: torch_segmentation_learn
=================================

Description:
    Trains a deep convolutional neural network (CNN) for segmentation.

Parameters:            
    - input_image (SMImage): Input image to be used for training (that include the reference mask as label_data).
        These are passed through by the `execute` method and are aggregated into the `results` arg of the `aggregate` method, where training is performed.
    - model_backbone (str): Currently supported models are defined in models/unet_zoo.py.
    - num_classes (int): number of class labels to be included in the segmentation prediction (typically 1).
    - learning_rate (float): Training parameter.
    - batch_size (int): Training parameter.
    - num_epochs (int): Training parameter.
    - split_ratio (tuple[float, float, float]): Training, validation, test split. Default = [6, 2, 2].
    - split_seed (int): Default = 42.
    - gpu_num: GPU to be used for computation. Default = 0, i.e., use CPU.
    - output_dir (str): Default = "../output".

Output:
    - None (weights file and log file outputs are not directly accessible to other tools).
            
Example JSON Plan:
  "neural_net-torch_seg": {
    "code": "torch_segmentation.py",
    "code_learn": "torch_segmentation_learn.py",
    "input_image": "from image_preprocessing-clahe",
    "model_backbone": "resnet50",
    "weights_path": "/home/matt/weights/right_lung_weights.pth",
    "num_classes": 1,
    "map_output_dir": "from arg output_dir",
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 100
  }
  
Notes:
    - Uses the pytorch package: https://docs.pytorch.org/docs/stable/
    - Online resource for basic understanding of training parameters: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    - torch_segmentation_learn writes weights and log files, but no output to the Blackboard, i.e., no output that can be used as input to other tools
```

## reasoning

### cand_select.py

```
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
```

### decision_tree.py

```
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
```

### decision_tree_learn.py

```
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
    - output_dir (str): Where training outputs will be written.
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
```
