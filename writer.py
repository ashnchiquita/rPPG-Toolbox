import os
import csv
import re

def natural_sort_key(s):
    # Split string into list of strings and integers for sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def write_naturally_sorted_csv(output_csv='output.csv'):
    current_dir = '/mnt/c/dataset/UBFC-rPPG/cache/UBFC-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceTrue_BackendHC_Large_boxFalse_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse_unsupervised'
    file_info = []

    for root, _, files in os.walk(current_dir):
        for file in files:
            if 'input' in file:  # Check if 'input' is in the filename
                full_path = os.path.join(root, file)
                file_info.append(full_path)

    # Natural sort by filename (not full path)
    file_info.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['idx', 'full_file_path'])
        for idx, path in enumerate(file_info):
            writer.writerow([idx, path])

# Run the function
write_naturally_sorted_csv()

