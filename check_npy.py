import numpy as np
import os
from pathlib import Path

def validate_input_label_lengths(directory_path="."):
    """
    Validate that input and label NPY files have matching lengths.
    
    Args:
        directory_path: Path to directory containing the NPY files
    
    Returns:
        dict: Results of validation for each prefix
    """
    # Get all NPY files in the directory
    npy_files = list(Path(directory_path).glob("*.npy"))
    
    # Extract prefixes from input files
    prefixes = set()
    for file in npy_files:
        if "_input0.npy" in file.name:
            prefix = file.name.replace("_input0.npy", "")
            prefixes.add(prefix)
    
    results = {}
    
    for prefix in sorted(prefixes):
        input_file = Path(directory_path) / f"{prefix}_input0.npy"
        label_file = Path(directory_path) / f"{prefix}_label0.npy"
        
        try:
            # Load the arrays
            input_data = np.load(input_file)
            label_data = np.load(label_file)
            
            # Check if lengths match
            input_length = len(input_data)
            label_length = len(label_data)
            lengths_match = input_length == label_length
            
            results[prefix] = {
                'input_length': input_length,
                'label_length': label_length,
                'lengths_match': lengths_match,
                'status': 'PASS' if lengths_match else 'FAIL'
            }
            
        except FileNotFoundError as e:
            results[prefix] = {
                'status': 'ERROR',
                'error': f"File not found: {e.filename}"
            }
        except Exception as e:
            results[prefix] = {
                'status': 'ERROR', 
                'error': str(e)
            }
    
    return results

def print_validation_results(results):
    """Print validation results in a readable format."""
    print("=" * 60)
    print("INPUT-LABEL LENGTH VALIDATION RESULTS")
    print("=" * 60)
    
    for prefix, result in results.items():
        print(f"\nPrefix: {prefix}")
        if result['status'] == 'ERROR':
            print(f"  Status: {result['status']}")
            print(f"  Error: {result['error']}")
        else:
            print(f"  Input length:  {result['input_length']:,}")
            print(f"  Label length:  {result['label_length']:,}")
            print(f"  Lengths match: {result['lengths_match']}")
            print(f"  Status: {result['status']}")

# Run the validation
if __name__ == "__main__":
    # Validate files in current directory
    validation_results = validate_input_label_lengths('/mnt/c/dataset/PURE/cache/PURE_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceTrue_BackendHC_Large_boxFalse_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_unsupervised')
    
    # Print results
    print_validation_results(validation_results)
    
    # Summary
    total_files = len(validation_results)
    passed = sum(1 for r in validation_results.values() 
                if r.get('status') == 'PASS')
    failed = sum(1 for r in validation_results.values() 
                if r.get('status') == 'FAIL')
    errors = sum(1 for r in validation_results.values() 
                if r.get('status') == 'ERROR')
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total prefixes: {total_files}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
