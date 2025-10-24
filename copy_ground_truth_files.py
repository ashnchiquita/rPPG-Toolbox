import os
import shutil
from pathlib import Path
import re

def copy_ground_truth_files(source_dir, destination_dir):
    """
    Copy all ground_truth.txt files from UBFC-rPPG dataset structure to a flat directory
    with numbered filenames.
    
    Args:
        source_dir (str): Path to the source directory (e.g., 'data/UBFC-rPPG/')
        destination_dir (str): Path to the destination directory (e.g., 'newfolder/')
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Counter for copied files
    copied_count = 0
    
    # Get all subject directories and sort them
    subject_dirs = []
    for item in source_path.iterdir():
        if item.is_dir() and item.name.startswith('subject'):
            subject_dirs.append(item)
    
    # Sort directories to ensure consistent numbering
    subject_dirs.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]) if re.findall(r'\d+', x.name) else 0)
    
    # Iterate through sorted subject directories
    for subject_dir in subject_dirs:
        ground_truth_file = subject_dir / "ground_truth.txt"
        
        if ground_truth_file.exists():
            # Extract subject number from directory name
            subject_number = re.findall(r'\d+', subject_dir.name)[0] if re.findall(r'\d+', subject_dir.name) else "unknown"
            # Copy the ground truth file with subject number as filename
            dest_file = dest_path / f"{subject_number}.txt"
            shutil.copy2(ground_truth_file, dest_file)
            print(f"Copied: {ground_truth_file} -> {dest_file}")
            copied_count += 1
        else:
            print(f"Warning: ground_truth.txt not found in {subject_dir.name}")
    
    print(f"\nTotal files copied: {copied_count}")

def main():
    # Configuration
    source_directory = "/mnt/f/chiquita/dataset/UBFC-rPPG/raw"
    destination_directory = "/mnt/d/dataset/UBFC-rPPG/gt-only"

    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Error: Source directory '{source_directory}' does not exist.")
        print("Please update the source_directory variable with the correct path.")
        return
    
    print(f"Copying ground truth files from '{source_directory}' to '{destination_directory}'...")
    copy_ground_truth_files(source_directory, destination_directory)
    print("Done!")

if __name__ == "__main__":
    main()
