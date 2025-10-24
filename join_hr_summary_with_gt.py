import pandas as pd
import os

def find_matching_gt_file(hr_summary_df, gt_dir):
    """
    Find the matching ground truth file based on the length of hr_summary_df.
    Returns the path to the matching ground truth CSV file.
    """
    chunk_lengths = [128, 160, 180]
    hr_summary_length = len(hr_summary_df)
    
    for chunk_length in chunk_lengths:
        gt_file_path = os.path.join(gt_dir, f"hr-{chunk_length}.csv")
        if os.path.exists(gt_file_path):
            gt_df = pd.read_csv(gt_file_path)
            if len(gt_df) == hr_summary_length:
                print(f"Found matching ground truth file: hr-{chunk_length}.csv (length: {len(gt_df)})")
                return gt_file_path
    
    print(f"Warning: No matching ground truth file found for length {hr_summary_length}")
    return None

def join_hr_summary_with_gt(hr_summary_file, gt_dir, output_dir):
    """
    Join HR summary file with the appropriate ground truth file.
    """
    # Read HR summary file
    hr_summary_df = pd.read_csv(hr_summary_file)
    print(f"Processing {os.path.basename(hr_summary_file)} with {len(hr_summary_df)} rows")
    
    # Find matching ground truth file
    gt_file_path = find_matching_gt_file(hr_summary_df, gt_dir)
    
    if gt_file_path is None:
        print(f"Skipping {hr_summary_file} - no matching ground truth file found")
        return
    
    # Read ground truth file
    gt_df = pd.read_csv(gt_file_path)
    
    # Standardize column names and data types for joining
    # HR summary uses 'subject_id', 'sort_index'
    # Ground truth uses 'Subject_id', 'Sort_index'
    hr_summary_df_copy = hr_summary_df.copy()
    
    # Rename columns in hr_summary to match ground truth format for joining
    if 'subject_id' in hr_summary_df_copy.columns:
        hr_summary_df_copy = hr_summary_df_copy.rename(columns={'subject_id': 'Subject_id'})
    if 'sort_index' in hr_summary_df_copy.columns:
        hr_summary_df_copy = hr_summary_df_copy.rename(columns={'sort_index': 'Sort_index'})
    
    # Extract numeric part from subject_id (e.g., "subject1" -> "1", "subject10" -> "10")
    if 'Subject_id' in hr_summary_df_copy.columns:
        hr_summary_df_copy['Subject_id'] = hr_summary_df_copy['Subject_id'].astype(str).str.replace('subject', '', regex=False)
    
    # Ensure data types match for joining
    # Convert Subject_id to string to handle mixed types, then ensure consistency
    if 'Subject_id' in hr_summary_df_copy.columns:
        hr_summary_df_copy['Subject_id'] = hr_summary_df_copy['Subject_id'].astype(str)
    if 'Subject_id' in gt_df.columns:
        gt_df['Subject_id'] = gt_df['Subject_id'].astype(str)
    
    # Ensure Sort_index is integer for both
    if 'Sort_index' in hr_summary_df_copy.columns:
        hr_summary_df_copy['Sort_index'] = hr_summary_df_copy['Sort_index'].astype(int)
    if 'Sort_index' in gt_df.columns:
        gt_df['Sort_index'] = gt_df['Sort_index'].astype(int)
    
    # Perform the join
    # Keep all columns from hr_summary and add Gt_hr from ground truth
    joined_df = hr_summary_df_copy.merge(
        gt_df[['Subject_id', 'Sort_index', 'Gt_hr']], 
        on=['Subject_id', 'Sort_index'], 
        how='left'
    )
    
    # Rename columns back to original hr_summary format
    if 'Subject_id' in joined_df.columns and 'subject_id' in hr_summary_df.columns:
        joined_df = joined_df.rename(columns={'Subject_id': 'subject_id'})
    if 'Sort_index' in joined_df.columns and 'sort_index' in hr_summary_df.columns:
        joined_df = joined_df.rename(columns={'Sort_index': 'sort_index'})
    
    # Create output file path
    output_file = os.path.join(output_dir, os.path.basename(hr_summary_file))
    
    # Save the joined data
    joined_df.to_csv(output_file, index=False)
    print(f"Saved joined data to {output_file} with {len(joined_df)} rows")
    
    # Check for any missing Gt_hr values
    missing_gt = joined_df['Gt_hr'].isna().sum()
    if missing_gt > 0:
        print(f"Warning: {missing_gt} rows have missing Gt_hr values")

def main():
    # Define paths
    hr_summary_dir = "/home/ashnchiquita/ta/rPPG-Toolbox/hr-summary"
    gt_dir = "/mnt/d/dataset/UBFC-rPPG/gt-split"
    output_dir = "/home/ashnchiquita/ta/rPPG-Toolbox/hr-summary-modified"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in hr-summary directory
    hr_summary_files = [f for f in os.listdir(hr_summary_dir) if f.endswith('.csv')]
    
    print(f"Found {len(hr_summary_files)} HR summary files to process")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Process each HR summary file
    for hr_summary_file in hr_summary_files:
        hr_summary_path = os.path.join(hr_summary_dir, hr_summary_file)
        try:
            join_hr_summary_with_gt(hr_summary_path, gt_dir, output_dir)
        except Exception as e:
            print(f"Error processing {hr_summary_file}: {str(e)}")
        print("-" * 30)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
