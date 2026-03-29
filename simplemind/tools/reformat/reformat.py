#!/usr/bin/env python3
"""
Reformat Script - PNG to NIfTI conversion with CSV creation.

This script converts PNG images to NIfTI format with proper orientation and creates a CSV file listing all converted images.

Usage:
    python reformat.py <input_folder>     # Convert all PNG images in folder to NIfTI and create CSV
"""

import numpy as np
import cv2
import nibabel as nib
from pathlib import Path
import argparse
import csv

def convert_png_to_nifti(png_path, output_nifti_path, reference_nifti_path=None):
    """
    Convert a PNG image to NIfTI format with proper orientation.
    
    Parameters:
        png_path (str): Path to the input PNG image
        output_nifti_path (str): Path for the output NIfTI file
        reference_nifti_path (str, optional): Reference NIfTI for header info
    
    Returns:
        tuple: (success, shape, message)
    """
    
    try:
        # Load PNG image
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False, None, f"Could not load image: {png_path}"
        
        # Keep as uint8 for efficient storage and consistency
        image_data = image.astype(np.uint8)
        
        # Apply 90° counter-clockwise rotation for proper medical imaging orientation
        image_data = np.rot90(image_data, k=1)
        
        # Create NIfTI image with uint8 datatype
        if reference_nifti_path and Path(reference_nifti_path).exists():
            # Use reference NIfTI header and affine
            ref_nii = nib.load(reference_nifti_path)
            nifti_img = nib.Nifti1Image(image_data, ref_nii.affine, ref_nii.header)
            # Ensure header reflects uint8 datatype
            nifti_img.header.set_data_dtype(np.uint8)
        else:
            # Create simple NIfTI with identity affine
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(image_data, affine)
            # Ensure header reflects uint8 datatype
            nifti_img.header.set_data_dtype(np.uint8)
        
        # Save NIfTI file
        nib.save(nifti_img, output_nifti_path)
        
        return True, image_data.shape, f"Successfully converted {Path(png_path).name}"
        
    except Exception as e:
        return False, None, f"Error converting {png_path}: {str(e)}"


def batch_convert_images(input_folder, output_folder=None):
    """
    Convert all PNG images in a folder to NIfTI format.
    
    Parameters:
        input_folder (str): Folder containing PNG images
        output_folder (str, optional): Folder to save converted NIfTI files. 
                                     If None, uses input folder name + '_nifti'
    
    Returns:
        tuple: (conversion_stats, converted_files_list)
    """
    
    input_folder = Path(input_folder)
    
    # Auto-generate output folder name if not provided
    if output_folder is None:
        output_folder = input_folder.parent / f"{input_folder.name}_nifti"
    else:
        output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    # Find all PNG files in input folder
    png_files = list(input_folder.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_folder}")
        return {"total": 0, "successful": 0, "failed": 0}, []
    
    print(f"Found {len(png_files)} images to convert")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Statistics
    successful_conversions = 0
    converted_files = []
    
    # Process each PNG file
    for i, png_file in enumerate(png_files, 1):
        print(f"=== Processing {i}/{len(png_files)}: {png_file.name} ===")
        
        # Generate output NIfTI path
        output_nifti = output_folder / f"{png_file.stem}.nii.gz"
        
        # Convert PNG to NIfTI
        success, shape, message = convert_png_to_nifti(str(png_file), str(output_nifti))
        
        if success:
            successful_conversions += 1
            print(f"✓ Converted: {message}")
            print(f"  Shape: {shape}")
            converted_files.append({
                'original_png': str(png_file),
                'nifti_file': str(output_nifti),
                'filename': png_file.name,
                'nifti_filename': f"{png_file.stem}.nii.gz"
            })
        else:
            print(f"✗ Failed: {message}")
        
        print()
    
    # Print summary
    print("=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(png_files)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {len(png_files) - successful_conversions}")
    print()
    
    if successful_conversions > 0:
        print(f"✓ {successful_conversions} images successfully converted to NIfTI format")
    
    return {
        "total": len(png_files),
        "successful": successful_conversions,
        "failed": len(png_files) - successful_conversions
    }, converted_files

def create_dataset_csv(converted_files, output_folder, csv_output_dir, csv_filename="dataset.csv"):
    """
    Create a CSV file listing all converted NIfTI images.
    
    Parameters:
        converted_files (list): List of converted file information
        output_folder (str): Folder where NIfTI images are stored (for path references)
        csv_output_dir (str): Directory where CSV file should be saved
        csv_filename (str): Name of the CSV file
    
    Returns:
        str: Path to the created CSV file
    """
    
    if not converted_files:
        print("No converted files to include in CSV")
        return None
    
    output_folder = Path(output_folder)
    csv_output_dir = Path(csv_output_dir)
    csv_path = csv_output_dir / csv_filename
    
    # Create CSV output directory if it doesn't exist
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CREATING DATASET CSV")
    print(f"{'='*60}")
    
    # Create CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header to match oct_images.csv format
        writer.writerow(['image', 'label_mask', 'sample_args'])
        
        # Write each converted file
        for file_info in converted_files:
            # Always use 'oct_images' as the folder name in the path
            image_path = f"oct_images/{file_info['nifti_filename']}"
            writer.writerow([
                image_path,                   # oct_images + NIfTI filename
                '',                           # Nothing goes in 'label_mask'
                '--upload_tags nifti'         # Same sample_args as oct_images.csv
            ])
            print(f"✓ Added to CSV: {image_path}")
    
    print(f"\n✓ CSV file created: {csv_path}")
    print(f"✓ Total entries: {len(converted_files)}")
    
    return str(csv_path)



def main():
    """Main function to convert PNG images to NIfTI format and create CSV."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert PNG images to NIfTI format and create dataset CSV")
    parser.add_argument("input_folder", help="Folder containing PNG images to convert")
    parser.add_argument("--output-folder", default=None, help="Output folder for NIfTI files and CSV (default: input_folder_name + '_nifti')")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("REFORMAT SCRIPT - PNG TO NIFTI CONVERSION & CSV CREATION")
    print("=" * 80)
    
    # Check if input folder exists
    if not Path(args.input_folder).exists():
        print(f"Error: Input folder '{args.input_folder}' not found!")
        return
    
    print("Converting PNG images to NIfTI format and creating dataset CSV")
    print("This will:")
    print("1. Convert all PNG images to NIfTI format with uint8 datatype")
    print("2. Apply 90° counter-clockwise rotation for proper orientation")
    print("3. Create a CSV file listing all converted images")
    print()
    
    # Determine output folder path
    input_folder = Path(args.input_folder)
    if args.output_folder is None:
        # Always save images to simplemind/data/oct_images
        actual_output_folder = Path("simplemind/data/oct_images")
    else:
        actual_output_folder = Path(args.output_folder)
    
    # Run batch conversion
    conversion_stats, converted_files = batch_convert_images(args.input_folder, actual_output_folder)
    
    # Create CSV file in simplemind/data (always named oct_images.csv)
    csv_path = None
    if converted_files:
        csv_filename = "oct_images.csv"
        csv_output_dir = Path("simplemind/data")
        csv_path = create_dataset_csv(converted_files, actual_output_folder, csv_output_dir, csv_filename)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print("PNG to NIfTI Conversion:")
    print(f"  Total images processed: {conversion_stats['total']}")
    print(f"  Successfully converted: {conversion_stats['successful']}")
    print(f"  Failed conversions: {conversion_stats['failed']}")
    print()
    
    if csv_path:
        print("Dataset CSV Creation:")
        print(f"  CSV file: {csv_path}")
        print(f"  Entries: {len(converted_files)}")
        print()
    
    print("Processing Status:")
    if conversion_stats['successful'] > 0:
        print("✅ Conversion completed successfully!")
        print(f"✅ {conversion_stats['successful']} images converted to NIfTI format")
        print(f"✅ Output saved to: {args.output_folder}")
        if csv_path:
            print(f"✅ Dataset CSV created: {Path(csv_path).name}")
    elif conversion_stats['failed'] > 0:
        print("⚠️  Some conversions failed. Check error messages above.")
    else:
        print("⚠️  No images were processed. Check input folder.")

if __name__ == "__main__":
    main()