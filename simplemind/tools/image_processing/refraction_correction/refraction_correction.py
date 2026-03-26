"""
Tool Name: refraction_correction
=================================

Description:
    Performs OCT refraction correction (dewarping) on optical coherence tomography images.
    This tool corrects for optical distortions in OCT images by detecting corneal boundaries
    and applying physics-based dewarping algorithms.

Parameters:            
    - input_image (SMImage): Input OCT image to be corrected.
    - output_size (tuple, optional): Output image size as (height, width). Default is (1769, 2165).
    - focus_distance (float, optional): Distance of focus in mm. Default is 1000000.
    - imaging_width (float, optional): Imaging width in mm. Default is 16.5.
    - imaging_depth (float, optional): Imaging depth in air in mm. Default is 13.4819861431871.
    - n_cornea (float, optional): Refractive index of cornea. Default is 1.39.
    - n_water (float, optional): Refractive index of water. Default is 1.34.

Output:
    - SMImage: The refraction-corrected (dewarped) OCT image.
            
Example JSON Plan:
    "image_processing-refraction_correction": {
        "code": "refraction_correction.py",
        "input_image": "from input_image",
        "output_size": [1769, 2165],
        "focus_distance": 1000000,
        "imaging_width": 16.5,
        "imaging_depth": 13.4819861431871,
        "n_cornea": 1.39,
        "n_water": 1.34
    }

Notes:
    - Designed specifically for OCT images with corneal structures
    - Uses physics-based ray tracing for accurate refraction correction
    - Automatically detects outer and inner corneal boundaries
"""

import asyncio  # For asynchronous programming support
import cv2  # OpenCV library for computer vision operations
import numpy as np  # NumPy for numerical array operations

from skimage import exposure, morphology  # Scikit-image for image processing operations
from scipy.interpolate import CubicSpline, RegularGridInterpolator  # Scipy for interpolation functions
from scipy.ndimage import binary_fill_holes  # Scipy for binary image operations
from skimage.morphology import remove_small_objects  # Remove small objects from binary images
from skimage.filters import gaussian  # Gaussian filtering
from scipy.signal import savgol_filter  # Savitzky-Golay filter for smoothing
from scipy.stats import linregress  # Linear regression for statistical analysis
from ruptures import Pelt  # Change point detection algorithm

from sm_sample_processor import SMSampleProcessor  # Base class for SimpleMind tools
from sm_image import SMImage  # SimpleMind image data structure
from sm_sample_id import SMSampleID  # SimpleMind sample identifier


# ------------------------------- Utilities mirroring the standalone -------------------------------

def _cv_minmax_u8(x2d: np.ndarray) -> np.ndarray:
    """OpenCV min–max normalize to 0..255 uint8 (exactly like the script)."""
    return cv2.normalize(x2d.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # Normalize array to 0-255 range and convert to uint8

def _rot270_flip_h(x2d: np.ndarray) -> np.ndarray:
    """Rotate 270° (90° CW), then horizontal flip — same as the script."""
    x2d = np.rot90(x2d, 3)  # Rotate image 270 degrees (3 * 90 degrees counterclockwise)
    x2d = np.fliplr(x2d)    # Flip image horizontally (left-right)
    return x2d              # Return the transformed image


# ------------------------------- Core pipeline (lifted from your script) -------------------------------
def OCT_OuterCornea(input_image):
    """
    Detects the outer corneal boundary in OCT images using image processing techniques.
    
    This function implements a multi-step algorithm:
    1. Image thresholding and morphological operations
    2. Column-wise analysis to find corneal regions
    3. Boundary tracing and smoothing
    4. Dynamic trimming to remove artifacts
    """
    def dynamically_trim_outercornea_wings(x, y, poly_degree=3, deviation_threshold=5.4):
        """Remove wing artifacts from corneal boundary by fitting polynomial and keeping central region."""
        poly_fit = np.polyfit(x, y, deg=poly_degree)
        y_fit = np.polyval(poly_fit, x)
        deviation = np.abs(y - y_fit)
        smooth_dev = savgol_filter(deviation, window_length=min(51, len(x) // 2 * 2 + 1), polyorder=2)
        central_mask = smooth_dev < deviation_threshold
        indices = np.where(central_mask)[0]
        if len(indices) < 10:
            return x, y
        start, end = indices[0], indices[-1]
        return x[start:end + 1], y[start:end + 1]

    # === IMAGE PREPROCESSING ===
    if input_image.ndim == 3:
        originalgray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        originalgray = input_image.copy()

    Rows, Columns = originalgray.shape
    originaladj = exposure.rescale_intensity(originalgray)
    _, BW = cv2.threshold(originaladj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    BW2 = morphology.remove_small_objects(BW.astype(bool), min_size=5000)
    BW2 = (BW2 * 255).astype(np.uint8)

    # === CORNEAL REGION DETECTION ===
    # Analyze central column to identify corneal and lens regions
    column_sum = np.sum(binary_fill_holes(BW2[:, int(Columns/2)-100:int(Columns/2)+100]), axis=1)
    smoothed_column_sum = np.convolve(column_sum, np.ones(5)/5, mode='same')
    binary_column_sum = np.zeros_like(smoothed_column_sum)
    binary_column_sum[smoothed_column_sum > 120] = 1
    
    # Find transitions to identify anatomical boundaries
    topcornea_candidates = np.where(np.diff(binary_column_sum, prepend=0) == 1)[0]
    endcornea_candidates = np.where(np.diff(binary_column_sum, prepend=0) == -1)[0]
    topcornea = topcornea_candidates[0] if topcornea_candidates.size > 0 else 0
    endcornea = endcornea_candidates[0] if endcornea_candidates.size > 0 else 0
    
    # Identify lens region if present
    column_sum_diff = np.diff(binary_column_sum, prepend=0)
    transition_indices = np.where(column_sum_diff == 1)[0]
    toplens_candidates = transition_indices[transition_indices > endcornea]
    toplens = toplens_candidates[0] if toplens_candidates.size > 0 else None
    
    # Clean up binary mask based on detected regions
    BW2[:topcornea, :] = 0
    if toplens is not None:
        BW2[endcornea+1:toplens, int(Columns/2)-50:int(Columns/2)+50] = 0
    else:
        BW2[endcornea+1:, int(Columns/2)-50:int(Columns/2)+50] = 0

    BW2 = (binary_fill_holes(BW2) & remove_small_objects(BW2.astype(bool), min_size=15000, connectivity=8)).astype(np.uint8)
    BW3 = binary_fill_holes(BW2).astype(np.uint8)

    # === BOUNDARY TRACING ===
    # Trace the outer corneal boundary column by column
    y_outer_Cornea = np.zeros(Columns, dtype=int)
    x_outer_Cornea = np.arange(Columns)
    for col_idx in range(Columns):
        column_profile = BW3[:, col_idx]
        transitions = np.where(np.convolve(column_profile, [1, 1, 1, 1], mode='valid') == 4)[0]
        y_outer_Cornea[col_idx] = transitions[0] + 1 if transitions.size > 0 else 0

    # === BOUNDARY SMOOTHING AND SEGMENTATION ===
    # Apply smoothing and use change point detection to identify boundary segments
    y_outer_Cornea_smt = savgol_filter(y_outer_Cornea, window_length=5, polyorder=2)
    y = -y_outer_Cornea_smt + np.max(y_outer_Cornea_smt)
    change_point_detector = Pelt(model='linear', min_size=2, jump=1)
    change_points = change_point_detector.fit(y.reshape(-1, 1)).predict(pen=10)

    change_points = np.array(change_points)
    istart = np.concatenate(([0], change_points[:-1]))
    istop = np.copy(change_points)
    istop[-1] = len(y) - 1

    # Fit linear segments and calculate slopes
    segment_slopes = []
    y2 = np.zeros_like(y)
    for s in range(len(istart)):
        segment_indices = np.arange(istart[s], istop[s] + 1)
        if len(segment_indices) < 2:
            continue
        slope, intercept, _, _, _ = linregress(segment_indices, y[segment_indices])
        y2[segment_indices] = slope * segment_indices + intercept
        segment_slopes.append(slope)

    # === BOUNDARY TRIMMING ===
    # Identify and trim irregular segments at the wings of the cornea
    mid_candidates = np.where(istart >= Columns // 2)[0]
    mid_seg = mid_candidates[0] - 1 if mid_candidates.size > 0 else 0

    slope_signs = np.sign(segment_slopes)
    slope_sign_changes = np.diff(slope_signs, prepend=0)
    irregular_segments = [i for i, slope in enumerate(segment_slopes) if abs(slope) < 0.01 or abs(slope) > 2]
    irregular_segment_right = min([i for i in irregular_segments if i > mid_seg], default=None)
    irregular_segment_left = max([i for i in irregular_segments if i < mid_seg], default=None)

    # Determine left and right trimming boundaries
    left_trim_candidates = []
    if np.any(slope_sign_changes[:mid_seg] == 2):
        left_trim_candidates.append(istart[np.where(slope_sign_changes[:mid_seg] == 2)[0][-1]])
    if irregular_segment_left is not None:
        left_trim_candidates.append(istart[irregular_segment_left + 1])
    left_trim_position = max(left_trim_candidates, default=None)
    left_trim_position = (left_trim_position + 5) if left_trim_position is not None else (x_outer_Cornea[0] + 21)

    right_trim_candidates = []
    if np.any(slope_sign_changes[mid_seg + 1:] == 2):
        right_trim_candidates.append(istart[np.where(slope_sign_changes[mid_seg + 1:] == 2)[0][0] + mid_seg])
    if irregular_segment_right is not None:
        right_trim_candidates.append(istart[irregular_segment_right])
    right_trim_position = min(right_trim_candidates, default=None)
    right_trim_position = (right_trim_position - 5) if right_trim_position is not None else (x_outer_Cornea[-1] - 21)

    # Extract trimmed boundary
    left_boundary_y = y_outer_Cornea[left_trim_position]
    right_boundary_y = y_outer_Cornea[right_trim_position]
    x_outer_Cornea = x_outer_Cornea[left_trim_position:right_trim_position+1]
    y_outer_Cornea = y_outer_Cornea[left_trim_position:right_trim_position+1]

    # === ARTIFACT REMOVAL ===
    # Remove large discontinuities in the boundary
    discontinuity_indices = np.where(np.abs(np.diff(y_outer_Cornea)) >= 30)[0]
    if discontinuity_indices.size > 0 and np.any(x_outer_Cornea[discontinuity_indices] < Columns // 2 - 500):
        left_discontinuities = discontinuity_indices[np.where(x_outer_Cornea[discontinuity_indices] < Columns // 2 - 500)]
        left_trim_index = np.max(left_discontinuities) + 1
        x_outer_Cornea = x_outer_Cornea[left_trim_index:]
        y_outer_Cornea = y_outer_Cornea[left_trim_index:]

    discontinuity_indices = np.where(np.abs(np.diff(y_outer_Cornea)) >= 30)[0]
    if discontinuity_indices.size > 0 and np.any(x_outer_Cornea[discontinuity_indices] > Columns // 2 + 500):
        right_discontinuities = discontinuity_indices[np.where(x_outer_Cornea[discontinuity_indices] > Columns // 2 + 500)]
        right_trim_index = np.min(right_discontinuities)
        x_outer_Cornea = x_outer_Cornea[:right_trim_index + 1]
        y_outer_Cornea = y_outer_Cornea[:right_trim_index + 1]

    # Final dynamic trimming to remove wing artifacts
    x_outer_Cornea, y_outer_Cornea = dynamically_trim_outercornea_wings(x_outer_Cornea, y_outer_Cornea)

    # Return comprehensive structure with all detected features
    ExtCorneaStruct = {
        'ycornea': y_outer_Cornea,
        'xcornea': x_outer_Cornea,
        'topcornea': topcornea,
        'endcornea': endcornea,
        'toplens': toplens,
        'BW': BW2,
        'rows': Rows,
        'columns': Columns,
        'originalgray': originalgray
    }
    return ExtCorneaStruct


def OCT_InnerCornea(ExtCorneaStruct):
    """
    Detects the inner corneal boundary using the outer boundary information.
    
    This function uses contour tracing from seed points to find the inner corneal surface,
    which represents the cornea-aqueous humor interface.
    """
    def dynamically_trim_innercornea_wings(x, y, poly_degree=3, curvature_threshold=0.01):
        """Remove wing artifacts from inner corneal boundary using curvature analysis."""
        poly = np.polyfit(x, y, poly_degree)
        y_fit = np.polyval(poly, x)
        y_smooth = savgol_filter(y_fit, window_length=51, polyorder=3)
        curvature = np.gradient(np.gradient(y_smooth))
        central_region = np.abs(curvature) < curvature_threshold
        indices = np.where(central_region)[0]
        if len(indices) < 2:
            return x, y
        start, end = indices[0], indices[-1]
        return x[start:end+1], y[start:end+1]

    def trace_from_seed(mask, seed_point, direction):
        """Trace contour from a seed point in specified direction (left or right)."""
        if direction == 'left':
            flipped_mask = cv2.flip(mask, 1)
            flipped_seed = (seed_point[0], mask.shape[1] - seed_point[1] - 1)
            contours, _ = cv2.findContours(flipped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return [], []

        def dist_to_seed(contour, seed):
            pts = contour[:, 0, :]
            return np.min(np.sqrt((pts[:, 1] - seed[0])**2 + (pts[:, 0] - seed[1])**2))

        # Find contour closest to seed point
        best_contour = min(contours, key=lambda c: dist_to_seed(c, flipped_seed if direction == 'left' else seed_point))
        rows, cols = best_contour[:, 0, 1], best_contour[:, 0, 0]
        
        # Start tracing from point closest to seed
        dists = np.sqrt((rows - (flipped_seed[0] if direction == 'left' else seed_point[0]))**2 +
                        (cols - (flipped_seed[1] if direction == 'left' else seed_point[1]))**2)
        start_idx = np.argmin(dists)
        rows, cols = np.roll(rows, -start_idx), np.roll(cols, -start_idx)

        # Filter contour points to ensure smooth tracing
        rows_final, cols_final = [rows[0]], [cols[0]]
        for r, c in zip(rows[1:], cols[1:]):
            vertical_change = r - rows_final[-1]
            horizontal_change = c - cols_final[-1]
            if abs(vertical_change) > 10:  # Skip large vertical jumps
                continue
            if horizontal_change > 0:  # Only move rightward
                rows_final.append(r); cols_final.append(c)
            if horizontal_change == 0:
                continue

        if direction == 'left':
            cols_final = [mask.shape[1] - c - 1 for c in cols_final]
        return cols_final, rows_final

    # === EXTRACT OUTER BOUNDARY INFORMATION ===
    y_outer = ExtCorneaStruct['ycornea']
    x_outer = ExtCorneaStruct['xcornea']
    BWimage = ExtCorneaStruct['BW']
    Rows = ExtCorneaStruct['rows']
    Columns = ExtCorneaStruct['columns']
    endcornea = ExtCorneaStruct['endcornea']
    toplens = ExtCorneaStruct['toplens']
    topcornea = ExtCorneaStruct['topcornea']

    # === PREPARE MASK FOR INNER BOUNDARY DETECTION ===
    mask = BWimage.copy()
    mask[:topcornea, :] = 0  # Remove everything above cornea
    mid_col = Columns // 2
    mask[endcornea+10:, mid_col-10:mid_col+10] = 0  # Clear central region below cornea
    mask = binary_fill_holes(mask).astype(np.uint8) * 255

    # === TRACE INNER BOUNDARY FROM SEED POINTS ===
    # Use seed points on either side of center to trace left and right portions
    seed_left = (endcornea + 5, mid_col - 30)
    seed_right = (endcornea + 5, mid_col + 30)
    x_left, y_left = trace_from_seed(mask, seed_left, 'left')
    x_right, y_right = trace_from_seed(mask, seed_right, 'right')

    # === COMBINE AND PROCESS TRACED BOUNDARIES ===
    x_combined = np.concatenate((x_left[::-1], x_right))  # Combine left (reversed) and right
    y_combined = np.concatenate((y_left[::-1], y_right))

    if len(x_combined) < 2:  # Fallback if tracing failed
        return {'ycornea': y_outer, 'xcornea': x_outer, 'endcornea': endcornea}

    # Remove duplicate x-coordinates and sort
    x_unique, idx = np.unique(x_combined, return_index=True)
    x_inner_Cornea = x_unique
    y_inner_Cornea = y_combined[idx]

    # Apply dynamic trimming to remove wing artifacts
    x_trimmed, y_trimmed = dynamically_trim_innercornea_wings(x_inner_Cornea, y_inner_Cornea)

    IntCorneaStruct = {'ycornea': x_trimmed*0 + y_trimmed, 'xcornea': x_trimmed, 'endcornea': endcornea}
    return IntCorneaStruct


def OuterDewarp(im_s, im_t, focus_distance, imaging_width, imaging_depth, n_cornea, target_width, target_height, outer_boundary_spline, ShowColors):
    """
    Performs outer cornea refraction correction using ray tracing physics.
    This corrects for optical distortion caused by light refraction at the air-cornea interface.
    """
    source_height, source_width = im_s.shape[:2]  # Get source image dimensions (height, width)
    focal_length_source = focus_distance/imaging_depth * source_height  # Calculate focal length in source image pixels
    focal_length_target = focus_distance/imaging_depth * target_height  # Calculate focal length in target image pixels
    angular_scale = np.arcsin(imaging_width / (2 * focus_distance)) / (source_width / 2)  # Calculate angular scaling factor

    # Create coordinate grids for target image
    target_x, target_y = np.meshgrid(np.arange(-target_width/2, target_width/2), np.arange(target_height/2, -target_height/2, -1))
    # Calculate initial ray angles and distances (before refraction correction)
    source_x = np.arctan2(target_x, focal_length_target - target_y) / angular_scale  # Convert target x to source angle
    source_y = focal_length_source - np.sqrt(target_x**2 + (focal_length_target - target_y)**2) * (source_height / target_height)  # Calculate source y position

    corrected_source_x = np.zeros_like(source_x)  # Initialize corrected x coordinates
    corrected_source_y = np.zeros_like(source_y)  # Initialize corrected y coordinates

    optimization_x = np.arange(-target_width/2, target_width/2)  # Create array of x positions for optimization

    # Process each row of the target image
    for row_idx in range(target_height):
        target_y_current = target_height/2 - row_idx  # Current y position in target image
        cornea_boundary_y = outer_boundary_spline(target_x[row_idx, :])  # Get cornea boundary y-coordinates for this row
        is_above_cornea = target_y_current >= cornea_boundary_y  # Determine which pixels are above the cornea
        direct_path_length = np.sqrt(target_x[row_idx,:]**2 + (focal_length_target - target_y_current)**2)  # Direct path length for pixels above cornea

        current_x_estimate = optimization_x.copy()  # Initialize optimization variables
        # Iterative optimization with decreasing step sizes
        for step_size in np.logspace(np.log10(30), np.log10(0.1), 5):
            # Create candidate positions around current estimate
            candidate_positions = np.array([current_x_estimate + shift * step_size for shift in range(-2, 3)])
            candidate_y_positions = outer_boundary_spline(candidate_positions)  # Get y-coordinates on cornea boundary

            # Calculate optical path lengths for each candidate
            air_path_length = np.sqrt(candidate_positions**2 + (focal_length_target - candidate_y_positions)**2) + focal_length_target * is_above_cornea  # Path in air
            cornea_path_length = np.sqrt((target_x[row_idx,:] - candidate_positions)**2 + (target_y_current - candidate_y_positions)**2) * n_cornea  # Path in cornea
            total_path_length = air_path_length + cornea_path_length  # Total optical path length
            total_path_length[2, :] = total_path_length[2, :] * (~is_above_cornea) + direct_path_length * is_above_cornea  # Use direct path for above-cornea pixels

            min_path_indices = np.argmin(total_path_length, axis=0)  # Find minimum path length for each pixel
            current_x_estimate += (min_path_indices - 2) * step_size  # Update position estimates

        # Calculate final corrected coordinates
        corrected_y = outer_boundary_spline(current_x_estimate) * (~is_above_cornea) + target_y_current * is_above_cornea  # Y-coordinate (cornea boundary or original)
        corrected_source_x[row_idx, :] = np.arctan2(current_x_estimate, focal_length_target - corrected_y) / angular_scale  # Convert to source x-coordinate
        corrected_source_y[row_idx, :] = focal_length_source - (np.min(total_path_length, axis=0) * source_height / target_height)  # Convert to source y-coordinate

    # Interpolate source image to target coordinates
    source_y_coords = np.arange(source_height); source_x_coords = np.arange(source_width)  # Create source coordinate arrays
    interpolator = RegularGridInterpolator((source_y_coords, source_x_coords), im_s[:, :, 1], bounds_error=False, fill_value=0)  # Create interpolator
    interp_coordinates = np.vstack([(source_height/2 - corrected_source_y).ravel(), (corrected_source_x + source_width/2).ravel()]).T  # Prepare coordinates for interpolation
    remapped_image = interpolator(interp_coordinates).reshape(target_height, target_width)  # Perform interpolation

    # Update target image with corrected values
    im_t[:, :, 1] = np.clip(remapped_image, 0, 255).astype(np.uint8)  # Green channel
    im_t[:, :, 0] = im_t[:, :, 1]  # Copy to red channel
    im_t[:, :, 2] = im_t[:, :, 1]  # Copy to blue channel
    return im_t, corrected_source_x, corrected_source_y  # Return corrected image and coordinate mappings


def InnerDewarp(im_s, im_t, focus_distance, imaging_width, imaging_depth, n_cornea, n_water, target_width, target_height, outer_boundary_spline, inner_boundary_spline, ShowColors):
    """
    Performs inner cornea refraction correction for the cornea-aqueous humor interface.
    This is more complex as it handles refraction through multiple tissue layers.
    """
    source_height, source_width = im_s.shape[:2]  # Get source image dimensions
    focal_length_source = (focus_distance/imaging_depth)*source_height  # Focal length in source pixels
    focal_length_target = (focus_distance/imaging_depth)*target_height  # Focal length in target pixels
    angular_scale = np.arcsin(imaging_width/(2*focus_distance))/(source_width/2)  # Angular scaling factor

    # Create coordinate grids for target image
    target_x, target_y = np.meshgrid(np.arange(-target_width/2, target_width/2), np.arange(target_height/2 - 1, -target_height/2 - 1, -1))
    source_x = np.arctan2(target_x, focal_length_target - target_y) / angular_scale
    source_y = focal_length_source - np.sqrt(target_x**2 + (focal_length_target - target_y)**2) * (source_height/target_height)

    # Initialize optimization variables for upper and lower boundaries
    upper_boundary_x = np.arange(-target_width/2, target_width/2)
    lower_boundary_x = np.copy(upper_boundary_x)
    
    # Determine starting row for processing (avoid regions above upper boundary)
    start_row = int(np.max([1, np.floor(np.min(target_height/2 - outer_boundary_spline(upper_boundary_x)))]))
    optimization_steps = 3
    step_sizes = np.logspace(np.log10(1.7), np.log10(0.1), 5)
    path_length_matrix = np.zeros(((2*optimization_steps - 1)**2, target_width))

    # Process each row of the target image
    for row_idx in range(start_row, target_height):
        target_y_current = target_height/2 - row_idx  # Current y position
        
        # Determine pixel regions relative to boundaries
        is_above_outer = target_y_current >= outer_boundary_spline(target_x[row_idx,:])
        is_between_boundaries = (target_y_current < outer_boundary_spline(target_x[row_idx,:])) & (target_y_current >= inner_boundary_spline(target_x[row_idx,:]))
        is_below_inner = target_y_current < inner_boundary_spline(target_x[row_idx,:])
        direct_path_above = np.sqrt(target_x[row_idx,:]**2 + (focal_length_target - target_y_current)**2)

        # Multi-step optimization for both upper and lower boundary intersections
        for step_size in step_sizes:
            for upper_step in range(2*optimization_steps - 1):
                upper_candidate_x = upper_boundary_x + (upper_step - optimization_steps + 1)*step_size
                upper_candidate_y = outer_boundary_spline(upper_candidate_x)
                upper_path_length = np.sqrt(upper_candidate_x**2 + (focal_length_target - upper_candidate_y)**2)
                between_boundaries_path = np.sqrt((target_x[row_idx,:]-upper_candidate_x)**2 + (target_y_current-upper_candidate_y)**2)*n_cornea
                path_through_cornea = upper_path_length + between_boundaries_path
                
                for lower_step in range(2*optimization_steps - 1):
                    matrix_idx = (2*optimization_steps-1)*lower_step + upper_step
                    lower_candidate_x = lower_boundary_x + (lower_step - optimization_steps + 1)*step_size
                    lower_candidate_y = inner_boundary_spline(lower_candidate_x)
                    boundary_to_boundary_path = np.sqrt((lower_candidate_x - upper_candidate_x)**2 + (lower_candidate_y - upper_candidate_y)**2)*n_cornea
                    aqueous_path_length = np.sqrt((target_x[row_idx,:] - lower_candidate_x)**2 + (target_y_current - lower_candidate_y)**2)*n_water
                    full_path_below = upper_path_length + boundary_to_boundary_path + aqueous_path_length
                    
                    # Select appropriate path based on pixel location
                    is_center_candidate = (upper_step == optimization_steps - 1)*(lower_step == optimization_steps - 1)
                    is_upper_center = (lower_step == optimization_steps - 1)
                    path_length_matrix[matrix_idx,:] = (is_above_outer*(is_center_candidate*direct_path_above + (1 - is_center_candidate)*5*focal_length_target) +
                                      is_between_boundaries*(is_upper_center*path_through_cornea + (1 - is_upper_center)*5*focal_length_target) +
                                      is_below_inner*full_path_below)

            # Update boundary estimates based on minimum path lengths
            min_path_indices = np.argmin(path_length_matrix, axis=0)
            upper_shift = min_path_indices % (2*optimization_steps - 1) - optimization_steps + 1
            lower_shift = min_path_indices // (2*optimization_steps - 1) - optimization_steps + 1
            upper_boundary_x += upper_shift*step_size
            lower_boundary_x += lower_shift*step_size

        # Calculate final corrected coordinates for this row
        corrected_y = outer_boundary_spline(upper_boundary_x)*(~is_above_outer) + (target_height/2 - row_idx)*is_above_outer
        corrected_source_x = np.arctan2(upper_boundary_x, focal_length_target - corrected_y)/angular_scale
        corrected_source_y = focal_length_source - np.min(path_length_matrix, axis=0)*(source_height/target_height)
        source_x[row_idx,:] = corrected_source_x
        source_y[row_idx,:] = corrected_source_y

    # Interpolate source image to corrected coordinates
    source_y_coords, source_x_coords = np.arange(source_height), np.arange(source_width)
    interpolator = RegularGridInterpolator((source_y_coords, source_x_coords), im_s[:,:,1].astype(float), bounds_error=False, fill_value=0)
    interpolation_coords = np.vstack([(source_height/2 - source_y).flatten(), (source_x + source_width/2).flatten()]).T
    interpolated_result = interpolator(interpolation_coords).reshape(target_height,target_width)

    # Update target image channels
    im_t[:,:,1] = np.clip(interpolated_result,0,255).astype(np.uint8)
    im_t[:,:,0] = im_t[:,:,1]
    im_t[:,:,2] = im_t[:,:,1]
    return im_t, source_x, source_y


# ------------------------------------ SimpleMind Tool wrapper ------------------------------------

class RefractionCorrection(SMSampleProcessor):
    """
    A tool that performs OCT refraction correction (dewarping) on optical coherence tomography images.
    """

    async def execute(
        self,
        *,
        input_image: SMImage,
        output_size: tuple = (1769, 2165),
        focus_distance: float = 1000000.0,
        imaging_width: float = 16.5,
        imaging_depth: float = 13.4819861431871,
        n_cornea: float = 1.39,
        n_water: float = 1.34,
        sample_id: SMSampleID
    ) -> SMImage:
        """
        Main execution function that performs OCT refraction correction.
        
        This function implements a complete pipeline:
        1. Load and preprocess the input image
        2. Detect outer and inner corneal boundaries
        3. Apply physics-based refraction correction
        4. Return the corrected image as SMImage
        """

        if input_image is None:  # Validate input
            self.print_error("Input image is None", sample_id)
            return None

        # Extract and validate parameters
        y_dimension, x_dimension = int(output_size[0]), int(output_size[1])  # Get target image dimensions

        # Get the image array and metadata from SMImage object
        src_array = input_image.pixel_array  # Extract numpy array from SMImage
        md = input_image.metadata or {}      # Get metadata dictionary (empty if None)

        # === IMAGE PREPROCESSING PHASE ===
        # Load and preprocess the image exactly like the standalone script
        original = self._load_like_standalone(src_array)  # Convert to RGB uint8 format with orientation correction
        originalgray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # Convert RGB to grayscale for processing

        # Initialize target image with fixed output dimensions
        im_t = np.zeros((y_dimension, x_dimension, 3), dtype=np.uint8)  # Create empty RGB target image
        originalgrayrsz = cv2.resize(originalgray, (x_dimension, y_dimension), interpolation=cv2.INTER_AREA)  # Resize to target dimensions

        # === CORNEAL BOUNDARY DETECTION PHASE ===
        # Detect outer and inner corneal boundaries using image processing
        Extcornea = OCT_OuterCornea(originalgrayrsz)  # Detect outer cornea boundary and extract features
        Intcornea = OCT_InnerCornea(Extcornea)        # Detect inner cornea boundary using outer boundary info

        # Extract boundary coordinates for spline fitting
        query_points = np.arange(1, Extcornea["xcornea"].shape[0] + 1)  # Create query points for spline evaluation
        x_outer_Cornea = Extcornea["xcornea"]  # X-coordinates of outer cornea boundary
        y_outer_Cornea = Extcornea["ycornea"]  # Y-coordinates of outer cornea boundary
        x_inner_Cornea = Intcornea["xcornea"]  # X-coordinates of inner cornea boundary
        y_inner_Cornea = Intcornea["ycornea"]  # Y-coordinates of inner cornea boundary

        # === BOUNDARY VALIDATION AND FITTING PHASE ===
        # Perform sanity checks on detected boundaries and create spline fits
        Somethingwrong = 0  # Flag to track if boundary detection failed

        # Validate outer cornea boundary shape
        x = x_outer_Cornea[::50]; y = y_outer_Cornea[::50]  # Sample every 50th point for efficiency
        polynomial_coefficients = np.polyfit(x, y, 4); y = np.polyval(polynomial_coefficients, query_points)     # Fit 4th degree polynomial
        idx = np.searchsorted(query_points, x); idx = np.clip(idx, 0, len(y)-1); y = y[idx]  # Map to query points
        outer_boundary_spline = CubicSpline(x - x_dimension/2, y_dimension/2 - y)          # Create cubic spline for outer boundary
        yout = (y_dimension/2) - outer_boundary_spline(query_points - x_dimension/2)                 # Evaluate spline at query points
        minpyout = np.argmin(yout)                          # Find minimum point (apex of cornea)
        signyout = np.sign(np.concatenate(([-1], np.diff(yout))))  # Check monotonicity
        if np.any(signyout[:minpyout] != -1) or np.any(signyout[minpyout + 1:] != 1):  # Validate shape
            Somethingwrong = 1  # Mark as invalid if not properly curved
            self.print_error("Boundary detection failed - image may not be suitable for dewarping", sample_id)
            return input_image # or return original image

        # Validate inner cornea boundary shape (similar process)
        x = x_inner_Cornea[::50]; y = y_inner_Cornea[::50]
        polynomial_coefficients = np.polyfit(x, y, 4); y = np.polyval(polynomial_coefficients, query_points)
        idx = np.searchsorted(query_points, x); idx = np.clip(idx, 0, len(y)-1); y = y[idx]
        inner_boundary_spline = CubicSpline(x - x_dimension/2, y_dimension/2 - y)
        yin = (y_dimension/2) - inner_boundary_spline(query_points - x_dimension/2)
        minpyin = np.argmin(yin)
        signyin = np.sign(np.concatenate(([-1], np.diff(yin))))
        if np.any(signyin[:minpyin] != -1) or np.any(signyin[minpyin + 1:] != 1):
            Somethingwrong = 1
            self.print_error("Boundary detection failed - image may not be suitable for dewarping", sample_id)
            return input_image # or return original image

        # Check that inner boundary is below outer boundary
        if np.any((yin - yout) <= 0):
            Somethingwrong = 1
            self.print_error("Boundary detection failed - image may not be suitable for dewarping", sample_id)
            return input_image # or return original image

        # === REFINED BOUNDARY FITTING PHASE ===
        # Create more robust 2nd-degree polynomial fits for the refraction correction
        
        # Fit outer cornea boundary with 2nd degree polynomial
        x_outer_sample = x_outer_Cornea[::50]; y_outer_sample = y_outer_Cornea[::50]  # Sample points
        xy_outer_sorted = sorted(zip(x_outer_sample, y_outer_sample), key=lambda t: t[0])  # Sort by x-coordinate
        x_outer_sorted, y_outer_sorted = np.array(xy_outer_sorted).T  # Separate x,y coordinates
        x_outer_unique, unique_idx = np.unique(x_outer_sorted, return_index=True)  # Remove duplicate x values
        y_outer_unique = y_outer_sorted[unique_idx]  # Get corresponding unique y values
        outer_polynomial_coefficients = np.polyfit(x_outer_unique, y_outer_unique, 2)  # Fit 2nd degree polynomial
        outer_boundary_spline = CubicSpline(x_outer_unique - x_dimension/2, y_dimension/2 - np.polyval(outer_polynomial_coefficients, x_outer_unique))  # Create spline from polynomial
        yout = (y_dimension/2) - outer_boundary_spline(query_points - x_dimension/2)  # Evaluate refined outer boundary

        # Fit inner cornea boundary with 2nd degree polynomial (similar process)
        x_inner_sample = x_inner_Cornea[::50]; y_inner_sample = y_inner_Cornea[::50]
        xy_inner_sorted = sorted(zip(x_inner_sample, y_inner_sample), key=lambda t: t[0])
        x_inner_sorted, y_inner_sorted = np.array(xy_inner_sorted).T
        x_inner_unique, unique_idx = np.unique(x_inner_sorted, return_index=True)
        y_inner_unique = y_inner_sorted[unique_idx]
        inner_polynomial_coefficients = np.polyfit(x_inner_unique, y_inner_unique, 2)
        inner_boundary_spline = CubicSpline(x_inner_unique - x_dimension/2, y_dimension/2 - np.polyval(inner_polynomial_coefficients, x_inner_unique))
        yin = (y_dimension/2) - inner_boundary_spline(query_points - x_dimension/2)
        
        # Final validation check
        if np.any((yin - yout) <= 0):
            Somethingwrong = 1
            self.print_error("Boundary detection failed - image may not be suitable for dewarping", sample_id)
            return input_image # or return original image

        # === REFRACTION CORRECTION PHASE ===
        # Apply physics-based refraction correction using the detected boundaries
        
        # Refraction dewarps (outer then inner) using provided parameters
        source_image = original  # Use RGB source image for correction
        
        # Step 1: Correct for outer cornea refraction (air-to-cornea interface)
        dewarpedOut, x_s1, y_s1 = OuterDewarp(source_image, im_t.copy(), focus_distance, imaging_width, imaging_depth, n_cornea, x_dimension, y_dimension, outer_boundary_spline, 1)
        
        # Step 2: Correct for inner cornea refraction (cornea-to-aqueous interface)
        dewarpedFull, x_s2, y_s2 = InnerDewarp(source_image, dewarpedOut.copy(), focus_distance, imaging_width, imaging_depth, n_cornea, n_water, x_dimension, y_dimension, outer_boundary_spline, inner_boundary_spline, 1)

        # === OUTPUT PREPARATION PHASE ===
        # Convert RGB to grayscale to eliminate the 3rd dimension
        dewarpedGray = cv2.cvtColor(dewarpedFull, cv2.COLOR_RGB2GRAY)  # Convert final result to grayscale

        # Update metadata for 2D image with corrected properties
        new_metadata = md.copy()  # Copy original metadata
        new_metadata.update({
            "direction": [1.0, 0.0, 0.0, 1.0],  # 2D direction matrix (2x2 flattened)
            "spacing": [1.0, 1.0],              # 2D spacing (isotropic)
            "origin": [0.0, 0.0],               # 2D origin at (0,0)
            "note": "refraction-corrected"      # Add processing note
        })

        self.print_log(f"Refraction correction complete. Output shape: {dewarpedGray.shape}", sample_id)

        # Return SimpleMind image with corrected data and updated metadata
        return SMImage(new_metadata, dewarpedGray, input_image.label_array)

    def _load_like_standalone(self, source) -> np.ndarray:
        """
        Load and preprocess image data with orientation correction.
        
        This function mimics the original standalone preprocessing:
        - Handles numpy arrays directly (primary use case)
        - Applies orientation correction for landscape images
        - Normalizes to uint8 range and converts to RGB
        
        Returns RGB uint8 (H, W, 3) format for downstream processing
        """
        if isinstance(source, np.ndarray):  # Handle numpy array input (most common case)
            vol = np.squeeze(source)        # Remove singleton dimensions
        else:
            # Handle file paths for raster images (PNG/JPG) - fallback case
            path = str(source)              # Convert to string path
            img = cv2.imread(path, cv2.IMREAD_COLOR)  # Load as color image
            if img is None:                 # Check if loading failed
                raise FileNotFoundError(f"Could not read image: {path}")
            return img                      # Return loaded image directly

        # Handle different array dimensionalities
        if vol.ndim == 3:                   # If 3D volume (multiple slices)
            z = vol.shape[0] - 1            # Get index of last slice
            slice2d = vol[z, :, :]          # Extract the last slice as 2D
        elif vol.ndim == 2:                 # If already 2D (single slice)
            slice2d = vol                   # Use as-is
        else:                               # If unsupported dimensionality
            raise ValueError(f"Unsupported array shape: {vol.shape}")

        slice2d = _cv_minmax_u8(slice2d)    # Normalize to 0-255 uint8 range

        # === ORIENTATION CORRECTION PHASE ===
        # Apply orientation correction for landscape images
        w, h = slice2d.shape                # Get width and height
        if w > h:                           # If landscape orientation (width > height)
            slice2d = _rot270_flip_h(slice2d)  # Apply 270° rotation + horizontal flip
        # else: portrait orientation → leave untouched

        # === RGB CONVERSION PHASE ===
        # Convert grayscale to RGB format for downstream processing
        slice3 = np.moveaxis([slice2d]*1, 0, -1)  # Create single-channel 3D array
        rgb = cv2.cvtColor(slice3, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        return rgb                          # Return RGB uint8 image


if __name__ == "__main__":   
    tool = RefractionCorrection()
    asyncio.run(tool.main())