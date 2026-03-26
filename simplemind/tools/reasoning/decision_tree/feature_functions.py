####### Notes for Developers ######
# When adding a feature, add it to the "Supported Features" list in docs/exercise3.md
# Feature functions must return a number or a boolean (which is interpreted as 0/1 by the decision tree)
# If an input mask is None then the function should return None
import sys
import numpy as np
from scipy.ndimage import binary_dilation

def calculate_centroid(roi_arr):
    if roi_arr is None:
        return None
    roi_arr = np.squeeze(roi_arr)
    if roi_arr.ndim == 2:
        nz = np.count_nonzero(roi_arr)
        if nz == 0:
            return None
        y_center, x_center = np.argwhere(roi_arr==1).sum(0)/nz
        return y_center, x_center
    elif roi_arr.ndim == 3:
        nz = np.count_nonzero(roi_arr)
        if nz == 0:
            return None
        z_center, y_center, x_center = np.argwhere(roi_arr==1).sum(0)/nz
        return z_center, y_center, x_center
    else:
        return None

def centroid_offset_x(roi_arr, roi_arr2, spacing=[1,1,1]): # Which one is roi_arr and roi_arr2
    if roi_arr is None or roi_arr2 is None:
        return None
    if roi_arr.shape != roi_arr2.shape:
        print(f"WARNING: centroid_offset_x roi shapes differ: {roi_arr.shape} vs {roi_arr2.shape}", file=sys.stderr, flush=True)
    # Calculate centroids first
    c1 = calculate_centroid(roi_arr)
    c2 = calculate_centroid(roi_arr2)
    if c1 is None or c2 is None:
        return None
    x1 = c1[-1]
    x2 = c2[-1]
    
    # Returned results: positive value --> Left # For DICOM
    #                   negative value --> right

    # Returned results: positive value --> right # For nifti
    #                   negative value --> left

    x_s = spacing[0]
    # print(f"DEBUG centroid_offset_x: x1={x1}, x2={x2}, x_s={x_s}", file=sys.stderr, flush=True)
    return (x1 - x2) * x_s

# TODO fix the redundancy with the args, put all these functions as methods in a class
def LeftOf(roi_arr, roi_arr2, spacing=[1,1,1]):
    if roi_arr is None or roi_arr2 is None:
        return None
    
    pos = centroid_offset_x(roi_arr = roi_arr, roi_arr2 = roi_arr2, spacing=spacing)
    return -(pos)
# Returns the opposite sign, if object is to the left, 
# value will be positive and confidence is compute correctly

def RightOf(roi_arr, roi_arr2, spacing=[1,1,1]):
    if roi_arr is None or roi_arr2 is None:
        return None
    
    pos = centroid_offset_x(roi_arr = roi_arr, roi_arr2 = roi_arr2, spacing=spacing)
    return pos
# Returns the same sign, if object is to the right, 
# value will be positive and confidence is compute correctly


def centroid_offset_y(roi_arr, roi_arr2, spacing = [1,1,1]):
    # Calculate centroids first

    if roi_arr is None or roi_arr2 is None:
        return None
    if roi_arr.shape != roi_arr2.shape:
        print(f"WARNING: centroid_offset_y roi shapes differ: {roi_arr.shape} vs {roi_arr2.shape}", file=sys.stderr, flush=True)
    c1 = calculate_centroid(roi_arr)
    c2 = calculate_centroid(roi_arr2)
    if c1 is None or c2 is None:
        return None
    if len(c1) < 2 or len(c2) < 2:
        return None
    y1 = c1[-2]
    y2 = c2[-2]
    
    # Returned results: positive value --> below
    #                   negative value --> above

    y_s = spacing[1]
    return (y1 - y2) * y_s

def PosteriorTo(roi_arr, roi_arr2, spacing=[1,1,1]):
    if roi_arr is None or roi_arr2 is None:
        return None
    
    pos = centroid_offset_y(roi_arr = roi_arr, roi_arr2 = roi_arr2, spacing=spacing)
    return -(pos)
# Returns the opposite sign, if object is posterior, 
# value will be positive and confidence is compute correctly

def AnteriorTo(roi_arr, roi_arr2, spacing=[1,1,1]):
    if roi_arr is None or roi_arr2 is None:
        return None
    
    pos = centroid_offset_y(roi_arr = roi_arr, roi_arr2 = roi_arr2, spacing=spacing)
    return pos
# Returns the same sign, if object is anterior to, 
# value will be positive and confidence is compute correctly


def volume(roi_arr, spacing = [1,1,1]):
    if roi_arr is None:
        return None
    roi_arr = np.squeeze(roi_arr)
    num_voxels = np.count_nonzero(roi_arr)
    x_s, y_s, z_s = spacing
    return num_voxels * x_s * y_s * z_s

def calculate_area(roi_arr, spacing = [1,1,1]):
    if roi_arr is None:
        return None
    roi_arr = np.squeeze(roi_arr)
    num_voxels = np.count_nonzero(roi_arr)
    if len(spacing) == 2:
        x_s, y_s = spacing
    else:
        x_s, y_s,_ = spacing # May fail for a (x, y) image
    return num_voxels * x_s * y_s

def area(roi_arr, spacing = [1,1,1]):
    return calculate_area(roi_arr, spacing)


def overlap_fraction(roi_arr, roi_arr2, spacing=[1,1,1]):
    if roi_arr is None or roi_arr2 is None:
        return None
    if type(roi_arr2) is not np.ndarray or type(roi_arr) is not np.ndarray:
        return None
    if roi_arr.shape != roi_arr2.shape:
        print(f"WARNING: overlap_fraction roi shapes differ: {roi_arr.shape} vs {roi_arr2.shape}", file=sys.stderr, flush=True)
    else:
        roi_arr = np.squeeze(roi_arr)
        roi_arr2 = np.squeeze(roi_arr2)
        # Find the % of arr1 that is inside arr2
        #overlap = get_overlap(roi_arr1, roi_arr2)
        arr1_voxels = len(np.argwhere(roi_arr))
        shared_voxels = len(np.argwhere(np.logical_and(roi_arr, roi_arr2)))
        return shared_voxels/arr1_voxels 

def in_contact_with(roi_arr, roi_arr2, spacing=[1,1,1]):
    if roi_arr is None or roi_arr2 is None:
        return None
    if roi_arr.shape != roi_arr2.shape:
        print(f"WARNING: in_contact_with roi shapes differ: {roi_arr.shape} vs {roi_arr2.shape}", file=sys.stderr, flush=True)
    roi_arr = np.squeeze(roi_arr)
    roi_arr2 = np.squeeze(roi_arr2)

    # dilate_by_1_voxel
    #dilated_arr = dilate_by_1_voxel(roi_arr)
    if roi_arr.ndim == 2:
        structure = np.ones((3, 3), dtype=np.bool_)
    elif roi_arr.ndim == 3:
        structure = np.ones((3, 3, 3), dtype=np.bool_)
    else:
        return None
    dilated_arr = binary_dilation(roi_arr, structure, iterations=1).astype(int)

    overlap = overlap_fraction(dilated_arr, roi_arr2)

    if overlap > 0:
        return True
    else:
        return False
    
#def get_overlap(arr1, arr2):
    # Find the % of arr1 that is inside arr2
#    arr1_voxels = len(np.argwhere(arr1))
#    shared_voxels = len(np.argwhere(np.logical_and(arr1, arr2)))
#    return shared_voxels/arr1_voxels 

# def get_overlap(arr1, arr2)
#     if arr1 is None or arr2 is None:
#         return None
#     sum_arr = arr1 + arr2 # Overlapping values will be two
#     arr1_points = len(np.where(arr1 == 1)[0]) # Number of voxels that are 1 in arr1
#     sum_arr_points =  len(np.where(sum_arr == 2)[0]) # Number of voxels that are shared between arr1 and arr2
#     return sum_arr_points/arr1_points # Shared voxels, divided by the total voxels shows the fraction of arr1 that is in arr2

#def dilate_by_1_voxel(arr):
#    structure = np.ones((3, 3, 3), dtype=np.bool_)
#    dilated_arr = binary_dilation(arr, structure, iterations=1).astype(int)
#    return dilated_arr
