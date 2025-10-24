import numpy as np
import json
import os
import pandas as pd

def read_hr_ubfc(bvp_file):
    """Reads a bvp signal file."""
    with open(bvp_file, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        hr = [float(x) for x in str1[1].split()]
    return np.asarray(hr)

def read_wave(bvp_file):
    """Reads a bvp signal file."""
    with open(bvp_file, "r") as f:
        labels = json.load(f)
        hr = [label["Value"]["pulseRate"]
                  for label in labels["/FullPackage"]]
    return np.asarray(hr)

def chunk(hrs, chunk_length):
    clip_num = hrs.shape[0] // chunk_length
    hr_clips = [hrs[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    return np.array(hr_clips)

def avg_chunk(splitted_hr):
    return np.mean(splitted_hr, axis=1)


ubfc_gt_dir = "/mnt/d/dataset/UBFC-rPPG/gt-only"
pure_gt_dir = "/mnt/d/dataset/PURE/gt-only"

ubfc_split_dir = "/mnt/d/dataset/UBFC-rPPG/gt-split"
pure_split_dir = "/mnt/d/dataset/PURE/gt-split"

chunk_lengths = [128, 160, 180]

# load ubfc ground truth
ubfc_files = os.listdir(ubfc_gt_dir)
ubfc_data = []  # Store tuples of (filename, hr_data)
for file in ubfc_files:
    bvp_file = os.path.join(ubfc_gt_dir, file)
    if bvp_file.endswith(".txt"):
        hr = read_hr_ubfc(bvp_file)
        subject_id = os.path.splitext(file)[0]  # Remove extension
        ubfc_data.append((subject_id, hr))

# Create output directory if it doesn't exist
os.makedirs(ubfc_split_dir, exist_ok=True)

for chunk_length in chunk_lengths:
    ubfc_split_path = f"{ubfc_split_dir}/hr-{chunk_length}.csv"
    
    # Prepare data for CSV
    csv_data = []
    
    for subject_id, hr_data in ubfc_data:
        # Chunk the HR data
        chunked_hr = chunk(hr_data, chunk_length)
        # Calculate average for each chunk
        avg_hr = avg_chunk(chunked_hr)
        
        # Add each chunk as a row
        for sort_index, gt_hr in enumerate(avg_hr):
            csv_data.append({
                'Subject_id': subject_id,
                'Sort_index': sort_index,
                'Gt_hr': gt_hr
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(ubfc_split_path, index=False)
    print(f"Saved UBFC HR data to {ubfc_split_path} with {len(csv_data)} rows")

# load pure ground truth
pure_files = os.listdir(pure_gt_dir)
pure_data = []  # Store tuples of (subject_id, hr_data)
for file in pure_files:
    wave_file = os.path.join(pure_gt_dir, file)
    if wave_file.endswith(".json"):
        hr = read_wave(wave_file)
        # Convert filename format: "01-01" -> "101", "10-01" -> "1001"
        base_filename = os.path.splitext(file)[0]  # Remove extension
        parts = base_filename.split("-")
        if len(parts) == 2:
            subject_id = parts[0] + parts[1]  # Concatenate without dash
        else:
            subject_id = base_filename  # Fallback if format is different
        pure_data.append((subject_id, hr))

# Create output directory if it doesn't exist
os.makedirs(pure_split_dir, exist_ok=True)

for chunk_length in chunk_lengths:
    pure_split_path = f"{pure_split_dir}/hr-{chunk_length}.csv"
    
    # Prepare data for CSV
    csv_data = []
    
    for subject_id, hr_data in pure_data:
        # Chunk the HR data
        chunked_hr = chunk(hr_data, chunk_length)
        # Calculate average for each chunk
        avg_hr = avg_chunk(chunked_hr)
        
        # Add each chunk as a row
        for sort_index, gt_hr in enumerate(avg_hr):
            csv_data.append({
                'Subject_id': subject_id,
                'Sort_index': sort_index,
                'Gt_hr': gt_hr
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(pure_split_path, index=False)
    print(f"Saved PURE HR data to {pure_split_path} with {len(csv_data)} rows")
    
    