#!/usr/bin/env python3
"""
Script to convert BraTS2020 H5 files to 3D NIfTI format.
This script processes 2D slices stored in H5 files and converts them to 3D NIfTI volumes.
"""

import os
import h5py
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re
from tqdm import tqdm

def analyze_h5_structure(h5_file_path):
    """Analyze the structure of an H5 file to understand its contents."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            print(f"Keys in {h5_file_path}:")
            for key in f.keys():
                data = f[key]
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                if hasattr(data, 'attrs'):
                    print(f"    Attributes: {dict(data.attrs)}")
    except Exception as e:
        print(f"Error analyzing {h5_file_path}: {e}")

def get_volume_slices(data_dir):
    """Group H5 files by volume ID and return organized structure."""
    volume_slices = defaultdict(list)
    
    # Get all H5 files
    h5_files = list(Path(data_dir).glob("*.h5"))
    
    for h5_file in h5_files:
        # Extract volume ID from filename (e.g., volume_10_slice_0.h5 -> volume_10)
        match = re.match(r'volume_(\d+)_slice_(\d+)\.h5', h5_file.name)
        if match:
            volume_id = match.group(1)
            slice_id = int(match.group(2))
            volume_slices[volume_id].append((slice_id, h5_file))
    
    # Sort slices within each volume
    for volume_id in volume_slices:
        volume_slices[volume_id].sort(key=lambda x: x[0])
    
    return volume_slices

def load_h5_data(h5_file_path):
    """Load data from H5 file and return numpy array."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Try common keys that might contain the image data
            possible_keys = ['image', 'data', 'slice', 'volume', 'image_data']
            
            for key in possible_keys:
                if key in f:
                    return np.array(f[key])
            
            # If no common keys found, try the first key
            if len(f.keys()) > 0:
                first_key = list(f.keys())[0]
                return np.array(f[first_key])
            
            raise ValueError(f"No data found in {h5_file_path}")
            
    except Exception as e:
        print(f"Error loading {h5_file_path}: {e}")
        return None

def create_3d_volume(volume_slices_data):
    """Combine 2D slices into a 3D volume."""
    if not volume_slices_data:
        return None
    
    # Get the shape of the first slice
    first_slice = volume_slices_data[0]
    if first_slice is None:
        return None
    
    slice_shape = first_slice.shape
    num_slices = len(volume_slices_data)
    
    # Handle different data shapes
    if len(slice_shape) == 3:  # 3D data: (240, 240, 4) - 4 modalities
        # Create 4D array: (num_slices, height, width, modalities)
        volume_4d = np.zeros((num_slices, slice_shape[0], slice_shape[1], slice_shape[2]), dtype=first_slice.dtype)
        
        for i, slice_data in enumerate(volume_slices_data):
            if slice_data is not None:
                volume_4d[i] = slice_data
        
        return volume_4d
    else:  # 2D data: (240, 240)
        # Create 3D array: (num_slices, height, width)
        volume_3d = np.zeros((num_slices, slice_shape[0], slice_shape[1]), dtype=first_slice.dtype)
        
        for i, slice_data in enumerate(volume_slices_data):
            if slice_data is not None:
                volume_3d[i] = slice_data
        
        return volume_3d

# The HDF5 'image' channels are ordered FLAIR, T1, T1ce, T2. The suffixes below do not follow that order,
# so *_FLAIR.nii.gz holds T2, *_T2 holds T1ce, *_T1 holds FLAIR and *_T1ce holds T1 (see common.py).
def save_as_nifti(volume_data, output_dir, volume_id, modalities=['T1', 'T1ce', 'T2', 'FLAIR']):
    """Save volume data as NIfTI files."""
    try:
        # Create NIfTI image
        # Assuming standard spacing for brain MRI (1mm isotropic)
        affine = np.eye(4)
        
        saved_files = []
        
        if len(volume_data.shape) == 4:  # 4D data: (slices, height, width, modalities)
            # Save each modality separately
            for modality_idx, modality in enumerate(modalities):
                # Extract single modality volume
                single_modality_volume = volume_data[:, :, :, modality_idx]
                
                # Create output filename
                output_filename = f"BraTS20_Training_{volume_id.zfill(3)}_{modality}.nii.gz"
                output_path = os.path.join(output_dir, output_filename)
                
                # Create NIfTI image
                nii_img = nib.Nifti1Image(single_modality_volume, affine)
                
                # Add metadata
                nii_img.header['descrip'] = f'BraTS2020 Volume {volume_id} - {modality}'
                nii_img.header['cal_max'] = float(single_modality_volume.max())
                nii_img.header['cal_min'] = float(single_modality_volume.min())
                
                # Save the file
                nib.save(nii_img, output_path)
                saved_files.append(output_path)
                
        else:  # 3D data: (slices, height, width)
            # Save as single modality
            output_filename = f"BraTS20_Training_{volume_id.zfill(3)}.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            
            nii_img = nib.Nifti1Image(volume_data, affine)
            nii_img.header['descrip'] = f'BraTS2020 Volume {volume_id}'
            nii_img.header['cal_max'] = float(volume_data.max())
            nii_img.header['cal_min'] = float(volume_data.min())
            
            nib.save(nii_img, output_path)
            saved_files.append(output_path)
        
        return saved_files
        
    except Exception as e:
        print(f"Error saving volume {volume_id}: {e}")
        return []

def main():
    # Paths
    base = Path(os.environ.get("BTS_BASE", Path(__file__).resolve().parents[2]))
    data_dir = base / "archive" / "BraTS2020_training_data" / "content" / "data"
    output_dir = base / "archive" / "3D Slices Sorted"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Analyzing H5 file structure...")
    
    # First, let's analyze a few H5 files to understand their structure
    sample_files = list(Path(data_dir).glob("*.h5"))[:3]
    for sample_file in sample_files:
        analyze_h5_structure(sample_file)
        print("-" * 50)
    
    print("\nGrouping slices by volume...")
    volume_slices = get_volume_slices(data_dir)
    
    print(f"Found {len(volume_slices)} unique volumes")
    
    # Show some statistics
    for volume_id in list(volume_slices.keys())[:5]:
        num_slices = len(volume_slices[volume_id])
        print(f"Volume {volume_id}: {num_slices} slices")
    
    print("\nConverting volumes to 3D NIfTI format...")
    
    successful_conversions = 0
    failed_conversions = 0
    
    # Process each volume
    for volume_id, slices_info in tqdm(volume_slices.items(), desc="Processing volumes"):
        try:
            # Load all slices for this volume
            volume_slices_data = []
            
            for slice_id, h5_file in slices_info:
                slice_data = load_h5_data(h5_file)
                volume_slices_data.append(slice_data)
            
            # Create 3D/4D volume
            volume_data = create_3d_volume(volume_slices_data)
            
            if volume_data is not None:
                # Save as NIfTI files
                saved_files = save_as_nifti(volume_data, output_dir, volume_id)
                
                if saved_files:
                    successful_conversions += 1
                    print(f"Successfully converted volume {volume_id} -> {len(saved_files)} files")
                    for saved_file in saved_files:
                        print(f"  Created: {os.path.basename(saved_file)}")
                else:
                    failed_conversions += 1
                    print(f"Failed to save volume {volume_id}")
            else:
                failed_conversions += 1
                print(f"Failed to create volume data for volume {volume_id}")
                
        except Exception as e:
            failed_conversions += 1
            print(f"Error processing volume {volume_id}: {e}")
    
    print(f"\nConversion complete!")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Total volumes processed: {successful_conversions + failed_conversions}")
    
    # Verify the expected count
    expected_count = 369
    if successful_conversions == expected_count:
        print(f"SUCCESS: Created {expected_count} volumes as expected!")
        print(f"Each volume contains 4 modalities (T1, T1ce, T2, FLAIR)")
        print(f"Total NIfTI files created: {successful_conversions * 4}")
    else:
        print(f"WARNING: Expected {expected_count} volumes, but created {successful_conversions}")
        print(f"Total NIfTI files created: {successful_conversions * 4}")

if __name__ == "__main__":
    main()
