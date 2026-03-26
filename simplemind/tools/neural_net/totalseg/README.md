
# TotalSegmentator Tool

This tool wraps the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) model for use inside the `sm-incubator` pipeline.
It enables organ segmentation on CT volumes and provides consistent outputs in several modes (full labelmap, multi-class subset, and binary union).

---

## Overview

**Location:** `tools/neural_net/totalseg/`  
**Main entry point:** `totalseg.py` (called automatically by the pipeline)  
**Environment file:** `env.yaml`  
**Core script:** `ts_main.py`  
**Label specification:** `labels_total.json` and `labels_total.tsv`

The tool reads a CT volume, calls the official TotalSegmentator under a clean conda environment, and produces segmentation maps in NIfTI format.  
It supports three main output modes and can return results for specific anatomical regions of interest (ROIs).


---

## Input / Output

| Item             | Description                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------- |
| **Input**        | A 3D CT NIfTI file (automatically provided by the pipeline)                                   |
| **Output**       | A NIfTI segmentation map stored under `../output/totalseg/<sample_id>/totalseg_result.nii.gz` |
| **Legend files** | `labels_total.tsv` (per sample, describing class names)               |

---

## Example Plan Configuration

### 1. Full multi-organ segmentation

```json
{
  "totalseg": {
    "code": "totalseg.py",
    "context": "./tools/neural_net/totalseg/",
    "input_image": "from input_image",

    "task": "total",
    "fast": false,

    "output_mode": "labelmap_raw",
    "output_dir": "../output",
    "final_output": true
  },

  "save_png_totalseg": {
    "code": "save_png.py",
    "context": "./tools/image_processing/save_png/",
    "input_image": "from input_image",
    "input_mask": "from totalseg",

    "mask_slice_axis": 0,
    "mask_none": false,
    "mask_alpha": 0.5,

    "filename": "totalseg_dense",
    "title": "TS: dense labelmap",
    "output_dir": "./samples"
  }
}
```

This runs **all 117 available organs** and writes a dense multi-class labelmap.
Each voxel value corresponds to an integer ID defined in `labels_total.json`.

---

### 2. Multi-class subset (selected organs)

```json
{
  "totalseg": {
    "code": "totalseg.py",
    "context": "./tools/neural_net/totalseg/",
    "input_image": "from input_image",

    "task": "total",
    "roi_subset": "kidney_right,kidney_left,liver",
    "fast": false,

    "output_mode": "multiclass_canon",
    "output_dir": "../output",
    "final_output": true
  },

  "save_png_totalseg": {
    "code": "save_png.py",
    "context": "./tools/image_processing/save_png/",
    "input_image": "from input_image",
    "input_mask": "from totalseg",

    "mask_slice_axis": 0,
    "mask_none": false,
    "mask_alpha": 0.5,

    "filename": "totalseg_kidneys_liver",
    "title": "TS: kidneys + liver",
    "output_dir": "./samples"
  }
}

```

This mode creates a multi-class map where each selected organ has a **unique canonical ID** (matching those in `labels_total.json`).

---

### 3. Binary union (0 = background, 1 = selected organs)

```json
{
  "totalseg": {
    "code": "totalseg.py",
    "context": "./tools/neural_net/totalseg/",
    "input_image": "from input_image",

    "task": "total",
    "roi_subset": "kidney_right,kidney_left",
    "fast": false,

    "output_mode": "binary",
    "output_dir": "../output",
    "final_output": true
  },

  "save_png_totalseg": {
    "code": "save_png.py",
    "context": "./tools/image_processing/save_png/",
    "input_image": "from input_image",
    "input_mask": "from totalseg",

    "mask_slice_axis": 0,
    "mask_none": false,
    "mask_alpha": 0.5,

    "filename": "totalseg_kidneys_union",
    "title": "TS: kidneys union",
    "output_dir": "./samples"
  }
}
```

This merges the chosen classes into a **single binary mask**.
A corresponding `labels_total.tsv` file is generated with the following structure:

```
value   meaning
0       background
1       kidney_right + kidney_left
```

---

## Run the plan
```
export PYTHONPATH="$PWD:$PYTHONPATH"
find smtool -type d -name '__pycache__' -prune -exec rm -rf {} +
find tools  -type d -name '__pycache__' -prune -exec rm -rf {} +
python run_plan.py plans/liver_ct --dataset_csv data/liver_ct_images.csv --addr bb-1.heph.com:8080
```

---

## Important Notes

1. **Use exact keys** from `labels_total.tsv` or `labels_total.json` —
   e.g. use `"liver"`, not numeric ID `44`.
   All keys are lowercase with underscores (e.g., `kidney_right`, `portal_vein_and_splenic_vein`).

2. The output voxel intensities always correspond to canonical IDs, consistent across modes.

3. The environment is isolated: all TotalSegmentator calls occur inside the tool-specific environment defined by `env.yaml`.

---

## TotalSegmentator Canonical Label Lookup

| ID  | Name                         |
| --- | ---------------------------- |
| 1   | adrenal_gland_left           |
| 2   | adrenal_gland_right          |
| 3   | aorta                        |
| 4   | atrial_appendage_left        |
| 5   | autochthon_left              |
| 6   | autochthon_right             |
| 7   | brachiocephalic_trunk        |
| 8   | brachiocephalic_vein_left    |
| 9   | brachiocephalic_vein_right   |
| 10  | brain                        |
| 11  | clavicula_left               |
| 12  | clavicula_right              |
| 13  | colon                        |
| 14  | common_carotid_artery_left   |
| 15  | common_carotid_artery_right  |
| 16  | costal_cartilages            |
| 17  | duodenum                     |
| 18  | esophagus                    |
| 19  | femur_left                   |
| 20  | femur_right                  |
| 21  | gallbladder                  |
| 22  | gluteus_maximus_left         |
| 23  | gluteus_maximus_right        |
| 24  | gluteus_medius_left          |
| 25  | gluteus_medius_right         |
| 26  | gluteus_minimus_left         |
| 27  | gluteus_minimus_right        |
| 28  | heart                        |
| 29  | hip_left                     |
| 30  | hip_right                    |
| 31  | humerus_left                 |
| 32  | humerus_right                |
| 33  | iliac_artery_left            |
| 34  | iliac_artery_right           |
| 35  | iliac_vena_left              |
| 36  | iliac_vena_right             |
| 37  | iliopsoas_left               |
| 38  | iliopsoas_right              |
| 39  | inferior_vena_cava           |
| 40  | kidney_cyst_left             |
| 41  | kidney_cyst_right            |
| 42  | kidney_left                  |
| 43  | kidney_right                 |
| 44  | liver                        |
| 45  | lung_lower_lobe_left         |
| 46  | lung_lower_lobe_right        |
| 47  | lung_middle_lobe_right       |
| 48  | lung_upper_lobe_left         |
| 49  | lung_upper_lobe_right        |
| 50  | pancreas                     |
| 51  | portal_vein_and_splenic_vein |
| 52  | prostate                     |
| 53  | pulmonary_vein               |
| 54  | rib_left_1                   |
| 55  | rib_left_2                   |
| 56  | rib_left_3                   |
| 57  | rib_left_4                   |
| 58  | rib_left_5                   |
| 59  | rib_left_6                   |
| 60  | rib_left_7                   |
| 61  | rib_left_8                   |
| 62  | rib_left_9                   |
| 63  | rib_left_10                  |
| 64  | rib_left_11                  |
| 65  | rib_left_12                  |
| 66  | rib_right_1                  |
| 67  | rib_right_2                  |
| 68  | rib_right_3                  |
| 69  | rib_right_4                  |
| 70  | rib_right_5                  |
| 71  | rib_right_6                  |
| 72  | rib_right_7                  |
| 73  | rib_right_8                  |
| 74  | rib_right_9                  |
| 75  | rib_right_10                 |
| 76  | rib_right_11                 |
| 77  | rib_right_12                 |
| 78  | sacrum                       |
| 79  | scapula_left                 |
| 80  | scapula_right                |
| 81  | skull                        |
| 82  | small_bowel                  |
| 83  | spinal_cord                  |
| 84  | spleen                       |
| 85  | sternum                      |
| 86  | stomach                      |
| 87  | subclavian_artery_left       |
| 88  | subclavian_artery_right      |
| 89  | superior_vena_cava           |
| 90  | thyroid_gland                |
| 91  | trachea                      |
| 92  | urinary_bladder              |
| 93  | vertebrae_c1                 |
| 94  | vertebrae_c2                 |
| 95  | vertebrae_c3                 |
| 96  | vertebrae_c4                 |
| 97  | vertebrae_c5                 |
| 98  | vertebrae_c6                 |
| 99  | vertebrae_c7                 |
| 100 | vertebrae_t1                 |
| 101 | vertebrae_t2                 |
| 102 | vertebrae_t3                 |
| 103 | vertebrae_t4                 |
| 104 | vertebrae_t5                 |
| 105 | vertebrae_t6                 |
| 106 | vertebrae_t7                 |
| 107 | vertebrae_t8                 |
| 108 | vertebrae_t9                 |
| 109 | vertebrae_t10                |
| 110 | vertebrae_t11                |
| 111 | vertebrae_t12                |
| 112 | vertebrae_l1                 |
| 113 | vertebrae_l2                 |
| 114 | vertebrae_l3                 |
| 115 | vertebrae_l4                 |
| 116 | vertebrae_l5                 |
| 117 | vertebrae_s1                 |

---

### Binary Output Lookup Example

When using:

```json
"output_mode": "binary",
"roi_subset": "kidney_right,kidney_left"
```

| Value | Meaning                    |
| ----- | -------------------------- |
| 0     | background                 |
| 1     | kidney_right + kidney_left |

---
