import os
import shutil
from pathlib import Path

def copy_json_files(source_dir, destination_dir):
    """
    Copy all JSON files from PURE dataset structure to a flat directory.
    
    Args:
        source_dir (str): Path to the source directory (e.g., 'data/PURE/')
        destination_dir (str): Path to the destination directory (e.g., 'newfolder/')
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Counter for copied files
    copied_count = 0
    
    # Iterate through all subdirectories in the source
    for item in source_path.iterdir():
        if item.is_dir():
            # Look for JSON file with the same name as the directory
            json_file = item / f"{item.name}.json"
            
            if json_file.exists():
                # Copy the JSON file to destination
                dest_file = dest_path / json_file.name
                shutil.copy2(json_file, dest_file)
                print(f"Copied: {json_file} -> {dest_file}")
                copied_count += 1
            else:
                print(f"Warning: JSON file not found for {item.name}")
    
    print(f"\nTotal files copied: {copied_count}")

def main():
    # Configuration
    source_directory = "/mnt/e/chiquita/dataset/PURE/raw"
    destination_directory = "/mnt/d/dataset/PURE/gt-only"
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Error: Source directory '{source_directory}' does not exist.")
        print("Please update the source_directory variable with the correct path.")
        return
    
    print(f"Copying JSON files from '{source_directory}' to '{destination_directory}'...")
    copy_json_files(source_directory, destination_directory)
    print("Done!")

if __name__ == "__main__":
    main()
